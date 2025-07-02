import torch
from einops import rearrange
from torchvision import transforms as T


def random_resized_crop(
    imgs: torch.Tensor,
    target_size: tuple[int],
    scale: tuple[int] = (0.8, 1.0),
    ratio: tuple[int] = (0.75, 4.0 / 3.10),
):
    # first n-3 dimensions are batch dimensions
    dim = imgs.shape[:-3]
    imgs = imgs.view(-1, *imgs.shape[-3:])

    if imgs.shape[-1] == 3:
        imgs = rearrange(imgs, "b h w c -> b c h w")

    imgs = T.RandomResizedCrop(target_size, scale=scale, ratio=ratio)(imgs)
    imgs = rearrange(imgs, "b c h w -> b h w c")
    imgs = imgs.view(*dim, *imgs.shape[-3:])
    return imgs


def resize(imgs: torch.Tensor, size: tuple[int]):
    dim = imgs.shape[:-3]
    imgs = imgs.view(-1, *imgs.shape[-3:])
    if imgs.shape[-1] == 3:
        imgs = rearrange(imgs, "b h w c -> b c h w")

    imgs = T.Resize(size)(imgs)
    imgs = rearrange(imgs, "b c h w -> b h w c")
    imgs = imgs.view(*dim, *imgs.shape[-3:])
    return imgs
