import torch
from einops import rearrange
from torchvision import transforms as T


def resize(imgs: torch.Tensor, size: tuple[int]):
    dim = imgs.shape[:-3]
    imgs = imgs.reshape(-1, *imgs.shape[-3:])
    if imgs.shape[-1] == 3:
        imgs = rearrange(imgs, "b h w c -> b c h w")

    imgs = T.Resize(size)(imgs)
    imgs = rearrange(imgs, "b c h w -> b h w c")
    imgs = imgs.reshape(*dim, *imgs.shape[-3:])
    return imgs
