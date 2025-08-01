from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing_extensions import Literal

from .base import BaseOutput, PretrainedConfig, PretrainedModel
from .modeling_mixin import LatentActionMixin
from .st import STBlock
from .transformer import (
    AttnBlock,
    MAPBlock,
    RMSNorm,
    get_1D_position_embeddings,
    get_2D_position_embeddings,
)
from .utils import (
    denormalize_images,
    hwc2chw,
    normalize_images,
    patching,
    random_resized_crop,
    resize,
    unpatching,
)
from .vq import VectorQuantizer2 as VectorQuantizer


def select_last_k(
    x: list[list], k: list[int], loffset: int = 0, roffset: int | None = None
) -> tuple[list[list], list[int]]:
    assert len(x) == len(k), "Length of x and k must be the same"
    roffset = roffset if roffset is None else -roffset
    ret = [x[i][-k[i] + loffset : roffset] for i in range(len(k))]
    return ret


def mask2len(mask: torch.Tensor | None, ref: torch.Tensor | None = None) -> list[int]:
    """Convert a mask to a list of lengths."""
    if mask is None:
        return [ref.shape[1]] * ref.shape[0] if ref is not None else []
    return mask.sum(dim=1).tolist() if mask is not None else []


@dataclass
class IgorForwardOutput(BaseOutput):
    reconstructions: torch.Tensor
    label: torch.Tensor
    action_tokens: torch.Tensor
    codebook_indices: torch.Tensor


class IgorConfig(PretrainedConfig):
    # Data config
    resolution: int = 224
    patch_size: int = 14
    in_channels: int = 3
    d_t: int = 8
    # Augmentation config
    augment_type: str = "resize_crop"
    augment_level: Literal["clip", "batch"] = "clip"
    random_crop_scale: list[float] = [0.8, 1.0]  # noqa: RUF012
    random_crop_ratio: list[float] = [0.75, 4.0 / 3.0]  # noqa: RUF012
    # Model config
    mlp_ratio: float = 4.0
    use_xformers: bool | None = None
    # Encoder config
    encoder_depth: int = 12
    encoder_embed_dim: int = 768
    encoder_n_heads: int = 8
    action_latent_dim: int = 128
    st_use_qk_norm: bool = True
    num_learned_tokens: int = 4
    map_heads: int = 24

    # Decoder config
    decoder_depth: int = 8
    decoder_embed_dim: int = 768
    decoder_n_heads: int = 16
    use_qk_norm: bool = True
    # VQ config
    n_codes: int = 32

    # Derived config
    grid_size: int | None = None
    embed_tokens: int | None = None

    def model_post_init(self, __context):
        assert self.resolution % self.patch_size == 0, (
            "Image resolution must be divisible by patch size"
        )
        self.grid_size = self.resolution // self.patch_size
        self.embed_tokens = self.grid_size**2


class IgorPretrainedModel(PretrainedModel):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig):
        super().__init__(config)

    @staticmethod
    def transformer_initializer(m: nn.Module) -> None:
        # Use `xavier_uniform` following Jax ViT
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _init_weights(self, module) -> None:
        self.transformer_initializer(module)

    def preprocess(self, clips: torch.Tensor, augment_type="resize_crop"):
        if clips.shape[-1] == 3:
            clips = hwc2chw(clips)
        if augment_type == "resize_crop":
            if self.config.augment_level == "clip":
                clips = torch.stack(
                    [
                        random_resized_crop(
                            clip,
                            size=self.config.resolution,
                            scale=self.config.random_crop_scale,
                            ratio=self.config.random_crop_ratio,
                        )
                        for clip in clips
                    ]
                )
            elif self.config.augment_level == "batch":
                clips = random_resized_crop(
                    clips,
                    size=self.config.resolution,
                    scale=self.config.random_crop_scale,
                    ratio=self.config.random_crop_ratio,
                )
        else:
            clips = resize(clips, size=self.config.resolution)
        return normalize_images(clips)

    def patching(self, images: torch.Tensor):
        return patching(images, self.config.patch_size)

    def unpatching(self, patches: torch.Tensor):
        # [B T (nh nw) (ph pw c)]
        return unpatching(
            patches,
            self.config.grid_size,
            self.config.patch_size,
            self.config.in_channels,
        )

    def patch2image(self, patches):
        images = self.unpatching(patches)
        return denormalize_images(images)


class IgorEncoder(IgorPretrainedModel):
    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        from .embed import PatchEmbed

        self.embed = PatchEmbed(
            config.resolution, config.patch_size, config.encoder_embed_dim
        )
        self.pe_spatial = nn.Parameter(
            torch.from_numpy(
                get_2D_position_embeddings(config.encoder_embed_dim, config.grid_size)
            )
            .float()
            .unsqueeze(0),
            requires_grad=False,
        )
        self.pe_temporal = nn.Parameter(
            torch.from_numpy(
                get_1D_position_embeddings(config.encoder_embed_dim, config.d_t)
            )
            .float()
            .unsqueeze(0),
            requires_grad=False,
        )
        self.layers = nn.ModuleList(
            [
                STBlock(
                    config.encoder_embed_dim,
                    config.encoder_n_heads,
                    d_s=config.embed_tokens,
                    d_t=config.d_t,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=0.0,
                    use_xformers=config.use_xformers,
                    enable_layernorm_kernel=False,
                    st_use_qk_norm=config.st_use_qk_norm,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = RMSNorm(config.encoder_embed_dim)
        self.map_block = MAPBlock(
            n_latents=config.num_learned_tokens,
            embed_dim=config.encoder_embed_dim,
            n_heads=config.map_heads,
            mlp_ratio=config.mlp_ratio,
            output_dim=config.action_latent_dim,
            do_rms_norm=True,
            do_swish_glu=True,
            qk_norm=False,
        )
        self.post_init()

    def init_weights(self):
        nn.init.xavier_uniform_(
            self.embed.proj.weight.data.view([self.embed.proj.weight.data.shape[0], -1])
        )
        return super().init_weights()

    def forward(self, clips: torch.Tensor, clip_len: list[int] | None = None):
        embedding = self.embed(clips)  # [B, T, N, D]
        embedding = embedding + self.pe_spatial  # [B, T, N, D]
        clip_len = (
            clip_len if clip_len is not None else [clips.shape[1]] * clips.shape[0]
        )

        embeddings = select_last_k(embedding, clip_len)  # [B, T, N, D]
        x = torch.cat(embeddings, dim=0)  # [sum(T_i), N, D]
        for idx, layer in enumerate(self.layers):
            x = layer(x, clip_len, tpe=self.pe_temporal if idx == 0 else None)

        x = self.norm(x)  # [sum(T_i), 256, 768]
        xs = list(torch.split(x, split_size_or_sections=clip_len, dim=0))
        xs = torch.cat([(x[1:] + x[:-1]) / 2.0 for x in xs], dim=0)  # mean pooling
        action = self.map_block(xs)  # [B, n, d]
        action = rearrange(action, "b n d -> b 1 (n d)")  # [B, 1, n * d]
        return action

    @torch.inference_mode()
    def idm(
        self,
        clips: torch.Tensor | list[torch.Tensor],
        mask: torch.Tensor | None = None,
        padding_side: str = "left",
    ):
        # clips: [B T C H W], mask indicates the valid frames, similar to language model's token mask
        # mask: [B T]
        # prepare the input
        if isinstance(clips, list):
            max_len = max([x.shape[1] for x in clips])
            if padding_side == "left":
                clips = [
                    F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in clips
                ]  # [B Tmax C H W]
            else:
                clips = [F.pad(x, (0, max_len - x.shape[1], 0, 0)) for x in clips]
            clips = torch.stack(clips, dim=0)  # [B T C H W]
        elif len(clips.shape) == 4:
            clips = clips.unsqueeze(0)  # [1 T C H W]
            mask = mask.unsqueeze(0) if mask is not None else None  # [1 T]
        assert len(clips.shape) == 5, "Input shape must be [B T C H W] or [T C H W]"
        mask = (
            torch.ones(*clips.shape[:2], device=clips.device).bool()
            if mask is None
            else mask
        )  # [B T]
        clips = self.preprocess(clips, augment_type="no")

        # Use a time window of d_t frames to compute the latent action
        t = clips.shape[1]
        if t <= self.config.d_t:
            clips = F.pad(clips, (0, 0, 0, self.config.d_t - t))  # [B d_t C H W]
            mask = F.pad(mask, (0, self.config.d_t - t))  # [B d_t]
            t = self.config.d_t
        for i in range(t - self.config.d_t + 1):
            clips_ = clips[:, i : i + self.config.d_t]  # [B d_t C H W]
            mask_ = mask[:, i : i + self.config.d_t]  # [B d_t]
            action = self.forward(clips_, mask_.sum(dim=1).tolist())
            if i == 0:
                actions = torch.split(
                    action,
                    split_size_or_sections=(mask_.sum(dim=1) - 1).tolist(),
                    dim=0,
                )  # list of [T 1 nd]
            else:
                new_actions = torch.split(
                    action,
                    split_size_or_sections=(mask_.sum(dim=1) - 1).tolist(),
                    dim=0,
                )  # list of [T 1 nd]
                new_action = [xi[-1].unsqueeze(0) for xi in new_actions]  # [1 1 nd]
                actions = [
                    torch.cat([x, y], dim=0) for x, y in zip(actions, new_action)
                ]  # [T+1 1 nd]
        return actions


class IgorDecoder(IgorPretrainedModel):
    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        from .embed import PatchEmbed

        self.embed = PatchEmbed(
            config.resolution, config.patch_size, config.decoder_embed_dim
        )
        self.pe = nn.Parameter(
            torch.from_numpy(
                get_2D_position_embeddings(config.decoder_embed_dim, config.grid_size)
            )
            .float()
            .unsqueeze(0),
            requires_grad=False,
        )
        self.action_embed = nn.Linear(
            config.action_latent_dim * config.num_learned_tokens,
            config.decoder_embed_dim,
        )
        self.layers = nn.ModuleList(
            [
                AttnBlock(
                    config.decoder_embed_dim,
                    config.decoder_n_heads,
                    config.mlp_ratio,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                    qk_norm=config.use_qk_norm,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.norm = RMSNorm(config.decoder_embed_dim)
        head_dim = config.patch_size**2 * config.in_channels
        self.pred_head = nn.Linear(config.decoder_embed_dim, head_dim)
        self.post_init()

    def init_weights(self):
        nn.init.xavier_uniform_(
            self.embed.proj.weight.data.view([self.embed.proj.weight.data.shape[0], -1])
        )
        return super().init_weights()

    @torch.inference_mode()
    def apply_latent_action(self, images: torch.Tensor, la: torch.Tensor):
        single_image = False
        if len(images.shape) == 3 or (len(images.shape) == 4 and images.shape[0] == 1):
            single_image = True
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            la = la.unsqueeze(0)
        images = self.preprocess(images, augment_type="no")
        x = self.fdm(la, images=images)
        x = self.patch2image(x)
        if single_image:
            return x.squeeze(0)
        return x

    def fdm(
        self,
        la: torch.Tensor,
        images: torch.Tensor | None = None,
        embeddings: torch.Tensor | None = None,
    ):
        # images: [B C H W], la: [B 1 nd]
        if embeddings is None:
            x = self.embed(images)  # [B N D]
        else:
            x = embeddings

        x = x + self.pe  # [B N D]
        x = torch.cat([self.action_embed(la), x], dim=1)  # [B N + 1 D]
        for layer in self.layers:
            x = layer(kv=x, q=x)
        x = self.norm(x)
        x = x[:, 1:]  # [B N D]
        output = self.pred_head(x)  # [B N D]
        return output

    def forward(
        self,
        clips: torch.Tensor,
        la: torch.Tensor,
        clip_len: list[int] | None = None,
    ):
        clip_len = (
            clip_len if clip_len is not None else [clips.shape[1]] * clips.shape[0]
        )
        images = select_last_k(clips, clip_len)
        embeddings = self.embed(images)  # [[T_i, N, D]] * B

        label = torch.cat(select_last_k(clips, clip_len, loffset=1))
        label = self.patching(label)  # [sum(T_i - 1), N, D]
        x = torch.cat(select_last_k(embeddings, clip_len, roffset=1))
        output = self.fdm(la=la, embeddings=x)
        return output, label


class IgorModel(IgorPretrainedModel, LatentActionMixin):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.encoder = IgorEncoder(config)
        self.vq = VectorQuantizer(
            config.n_codes,
            config.action_latent_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=False,
        )
        self.decoder = IgorDecoder(config)
        self.post_init()

    def init_weights(self):
        nn.init.uniform_(
            self.vq.embedding.weight, -1.0 / self.vq.n_e, 1.0 / self.vq.n_e
        )
        return super().init_weights()

    def loss(
        self, targets: torch.Tensor, reconstructions: torch.Tensor
    ) -> torch.Tensor:
        mse = (reconstructions - targets) ** 2
        recon_loss = mse.mean()
        return recon_loss

    @torch.inference_mode()
    def apply_latent_action(self, images: torch.Tensor, la: torch.Tensor):
        return self.decoder.apply_latent_action(images, la)
