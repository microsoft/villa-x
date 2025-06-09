import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from lam import AutoModelForLatentAction
from udl.utils.rotation import euler_to_rotation_6d


def transform_state(state: torch.Tensor):
    xyz = state[..., :3]
    euler = state[..., 3:6]
    gripper = state[..., 6:]
    # rot = quaternion_to_rotation_6d(euler)
    rot = euler_to_rotation_6d(euler)
    assert len(xyz.shape) == 3 and xyz.shape[0] == 1 and xyz.shape[1] == 1
    rot = rot[None, None, :]
    return torch.cat([xyz, rot, gripper], dim=-1)


def normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
    eps: float = 1e-8,
    ignore_last: bool = True,
) -> np.ndarray:
    if ignore_last:
        ndata = (
            2
            * (data[..., :-1] - data_min[..., :-1])
            / (data_max[..., :-1] - data_min[..., :-1] + eps)
            - 1
        )
        ndata = torch.cat([ndata, data[..., -1:]], dim=-1)
    else:
        assert False, "should omit the last dimension"
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


def denormalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
    eps=1e-8,
) -> np.ndarray:
    clip_range = clip_max - clip_min
    rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
    return rdata


def normalize_gaussian(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    return (data - mean) / (std + eps)


def denormalize_gaussian(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    return data * (std + eps) + mean


class LAM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.use_continuous = config["use_continuous"]

        ckpt_dir = config["ckpt_dir"]
        self.model = AutoModelForLatentAction.from_pretrained(ckpt_dir)
        self.model = self.model.eval()

        self.n_latents = self.model.config.num_learned_tokens
        self.n_dim = self.model.config.action_latent_dim

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor, use_vq: bool = True):
        out = self.model.idm(imgs, return_dict=True)
        B, T, *_ = imgs.shape
        if not use_vq:
            return torch.stack(out["tokens"]).reshape(
                B, (T - 1) * self.n_latents, self.n_dim
            )
        if self.use_continuous:
            return out["vq_tokens"].reshape(B, (T - 1) * self.n_latents, self.n_dim)
        return out["indices"].reshape(B, (T - 1) * self.n_latents)

