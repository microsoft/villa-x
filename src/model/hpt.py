from typing import Optional

import torch
import torchvision
from einops import rearrange, repeat
from torch import einsum, nn
from udl.utils.preprocessing import resize as resize_image

INIT_CONST = 0.02
global_vision_model = None


def normalize_image(image: torch.Tensor, resize: bool = True) -> torch.Tensor:
    if resize:
        image = resize_image(image, (224, 224))

    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    image = image / 255.0
    image = (image - mean) / std
    return image


def unnormalize_image(image: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    image = image * std + mean
    image = image * 255
    image = image.clamp(0, 255).byte()
    return image


class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(
        self, query_dim: int, heads: int, dim_head: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(
        self,
        tokens: int,
        query_dim: int,
        out_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        """initialize cross attention module and the learnable tokens"""
        self.tokens = nn.Parameter(torch.randn(1, tokens, query_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            query_dim=query_dim, heads=heads, dim_head=dim_head, out_dim=out_dim, dropout=dropout
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)

        # stem_feat = self(x)
        # stem_feat = stem_feat.reshape(
        #     stem_feat.shape[0], -1, stem_feat.shape[-1]
        # )  # (32, 147, 128)
        # # Replicating tokens for each item in the batch and computing cross-attention
        # stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        # stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)

        stem_feat = self(x)  # [B, 512, 7, 7]
        stem_feat = rearrange(stem_feat, 'B D H W -> B (H W) D')
        stem_tokens = self.tokens.repeat(stem_feat.shape[0], 1, 1)  # (B, 16, 512)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (B, 16, 1024)

        return stem_tokens


class MLP(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: list[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*modules) for _ in range(num_of_copy)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                _input = x[:, idx]
                net = self.net[idx]
                out.append(net(_input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class ResNet(PolicyStem):
    def __init__(
        self, weights: str = "DEFAULT", resnet_model: str = "resnet18"
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])  # remove avgpoll & fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        # B, *_, H, W = x.shape
        # x = x.view(len(x), -1, 3, H, W)
        # return self.net(x)  # (B, T*N, 512, H/32, W/32)
        assert len(x.shape) == 4 and x.shape[1] == 3  # [B, 3, H, W]
        return self.net(x)  # [B, 512, H/32, W/32)


@torch.no_grad()
def get_resnet_embeddings(image, per_token=False, device="cuda", downsample=False):
    """Get Resnet embedding. Input: H x W x 3"""
    global global_vision_model

    if global_vision_model is None:
        global_vision_model = ResNet().to(device)
    device = global_vision_model.device

    image = normalize_image(image)
    global_vision_model.eval()
    image_th = torch.FloatTensor(image).to(device)
    if len(image_th.shape) == 3:
        image_th = image_th[None]
    image_th = rearrange(image_th, "... H W C -> ... C H W")

    # forward pass through encoder only
    output = global_vision_model.net(image_th)
    if downsample:  # pool to 3 x 3
        output = torch.nn.functional.avg_pool2d(output, 2, 2)

    output = output.reshape(output.shape[0], 512, -1).transpose(1, 2)
    return output