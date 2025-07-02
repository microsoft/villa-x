"""
transformer.py

General Transformer modules & utilities.

References:
    - https://github.com/facebookresearch/mae
    - https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Helper/Utility Function -- computes simple 1D sinusoidal position embeddings for both 1D/2D use cases.
#   > We'll be combining two 1D sin-cos (traditional) position encodings for height/width of an image (grid features).
def get_1D_sine_cosine(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.einsum(
        "m,d->md", pos.reshape(-1), omega
    )  # [flatten(pos) x omega] -- outer product!
    emb_sin, emb_cos = np.sin(out), np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [flatten(pos) x D]


# 1D Sine-Cosine Position Embedding -- standard from "Attention is all you need!"
def get_1D_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return get_1D_sine_cosine(embed_dim, np.arange(length))


# 2D Sine-Cosine Position Embedding (from MAE repository)
#   > https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2D_position_embeddings(
    embed_dim: int, grid_size: int, cls_token: bool = False, query_tokens: int = 0
) -> np.ndarray:
    # Create 2D Position embeddings by taking cross product of height and width and splicing 1D embeddings...
    grid_h, grid_w = (
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
    )
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(
        2, 1, grid_size, grid_size
    )  # w goes first?

    # Use half of dimensions to encode grid_h, other half to encode grid_w
    emb_h, emb_w = (
        get_1D_sine_cosine(embed_dim // 2, grid[0]),
        get_1D_sine_cosine(embed_dim // 2, grid[1]),
    )
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    # CLS token handling (only for R-MVP)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    if query_tokens > 0:
        pos_embed = np.concatenate(
            [pos_embed, np.zeros([query_tokens, embed_dim])], axis=0
        )

    return pos_embed


# Patch Embedding Module
class PatchEmbed(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int = 3,
        flatten: bool = True,
    ):
        super().__init__()
        self.resolution, self.patch_size = (
            (resolution, resolution),
            (patch_size, patch_size),
        )
        self.grid_size = (
            self.resolution[0] // self.patch_size[0],
            self.resolution[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.proj(patches)
        if self.flatten:
            return rearrange(
                patch_embeddings,
                "bsz embed patch_h patch_w -> bsz (patch_h patch_w) embed",
            )
        return patch_embeddings


# LayerScale -- Trainable scaling for residual blocks -- Mistral/CaIT
class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: float = 0.1
    ) -> None:  # CaIT :: 0.1 -> lay 12, 1e-5 -> lay 24, 1e-6...
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class FlashAttention(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, dropout: float = 0.0, qk_norm=False
    ) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"

        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5
        self.attn_softmax = None
        self.qk_norm = qk_norm

        # Projections
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if qk_norm:
            per_head_layernorm = embed_dim // n_heads
            self.q_layernorm = nn.LayerNorm(per_head_layernorm, elementwise_affine=True)
            self.k_layernorm = nn.LayerNorm(per_head_layernorm, elementwise_affine=True)

    def forward(
        self,
        q_in,
        kv_in,
    ) -> torch.Tensor:
        B_x, N_x, C_x = q_in.shape
        B_c, N_c, C_c = kv_in.shape

        assert B_x == B_c and C_x == C_c

        B, C = B_x, C_x

        # Project to Q-K-V
        kv = (
            self.kv(kv_in)
            .reshape(B, N_c, 2, self.n_heads, C // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = (
            self.q(q_in)
            .reshape(B, N_x, 1, self.n_heads, C // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q = q[0]

        if self.qk_norm:
            # q: [B, n_heads, N_1, emb_dim // n_heads]
            # k: [B, n_heads, N_2, emb_dim // n_heads]
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        vals = F.scaled_dot_product_attention(q, k, v)
        vals = vals.transpose(1, 2).reshape(B, N_x, C)

        # Project back to `embed_dim` -- with optional dropout
        vals = self.dropout(self.proj(vals))

        return vals


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        output_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
        qk_norm: bool = False,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.embed_dim, self.n_heads = embed_dim, n_heads
        self.n_latents = n_latents

        # Projection Operator
        self.pre_projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )

        self.attn = FlashAttention(
            self.embed_dim, n_heads=n_heads, dropout=dropout, qk_norm=qk_norm
        )

        # Position-wise Feed-Forward Components
        self.mlp_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(
                    nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)),
                    nn.GELU(),
                )
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

        # self.final_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, 2 * self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * self.embed_dim, output_dim),
        # )
        self.final_proj = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(
            latents + self.attn(q_in=latents, kv_in=self.pre_projection(x))
        )
        latents = self.mlp_norm(latents + self.mlp(latents))
        latents = latents.squeeze(dim=1)
        latents = self.final_proj(latents)
        return latents


class MAPBlock_2head(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        output_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
        qk_norm: bool = False,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.embed_dim, self.n_heads = embed_dim, n_heads
        self.n_latents = n_latents

        # Projection Operator
        self.pre_projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )

        self.attn = FlashAttention(
            self.embed_dim, n_heads=n_heads, dropout=dropout, qk_norm=qk_norm
        )

        # Position-wise Feed-Forward Components
        self.mlp_norm = (
            RMSNorm(self.embed_dim)
            if do_rms_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-6)
        )
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(
                    nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)),
                    nn.GELU(),
                )
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

        # self.final_proj = nn.Sequential(
        #     nn.Linear(self.embed_dim, 2 * self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * self.embed_dim, output_dim),
        # )
        self.final_proj_mu = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
        )
        self.final_proj_var = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(
            latents + self.attn(q_in=latents, kv_in=self.pre_projection(x))
        )
        latents = self.mlp_norm(latents + self.mlp(latents))
        latents = latents.squeeze(dim=1)
        latents_mu = self.final_proj_mu(latents)
        latents_var = self.final_proj_var(latents)
        return latents_mu, latents_var
