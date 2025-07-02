import torch
from einops import rearrange
from torchvision import transforms as T


def flatten_internal(fn, flatten_ndim=3):
    def wrapper(x: torch.Tensor, *args, **kwargs):
        dim = x.shape[:-flatten_ndim]
        x = x.reshape(-1, *x.shape[-flatten_ndim:])  # [B ABC]
        x = fn(x, *args, **kwargs)  # [B CD]
        x = x.reshape(*dim, *x.shape[-x.ndim + 1 :])
        return x

    return wrapper


@flatten_internal
def random_resized_crop(
    imgs: torch.Tensor,
    size: tuple[int] | int,
    scale: tuple[int] = (0.8, 1.0),
    ratio: tuple[int] = (0.75, 4.0 / 3.10),
):
    if isinstance(size, int):
        size = (size, size)
    imgs = T.RandomResizedCrop(size, scale=scale, ratio=ratio)(imgs)
    return imgs


@flatten_internal
def resize(imgs: torch.Tensor, size: tuple[int] | int):
    if isinstance(size, int):
        size = (size, size)
    imgs = T.Resize(size)(imgs)
    return imgs


def hwc2chw(imgs: torch.Tensor):
    return rearrange(imgs, "... h w c -> ... c h w")


def chw2hwc(imgs: torch.Tensor):
    return rearrange(imgs, "... c h w -> ... h w c")


def hwc_internal(fn):
    def wrapper(imgs, *args, **kwargs):
        imgs = chw2hwc(imgs)
        imgs = fn(imgs, *args, **kwargs)
        imgs = hwc2chw(imgs)
        return imgs

    return wrapper


@hwc_internal
def normalize_images(imgs: torch.Tensor, norm_type="imagenet"):
    if norm_type == "default":
        # put pixels in [-1, 1]
        return imgs.to(torch.float32) / 127.5 - 1.0
    elif norm_type == "imagenet":
        imgs = imgs.to(torch.float32) / 255
        assert imgs.shape[-1] % 3 == 0, "images should have rgb channels!"
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        return (imgs - mean) / std

    raise ValueError()


@hwc_internal
def denormalize_images(imgs: torch.Tensor, norm_type="imagenet"):
    if norm_type == "default":
        # put pixels in [0, 255]
        ret = (imgs + 1.0) * 127.5
    elif norm_type == "imagenet":
        assert imgs.shape[-1] % 3 == 0, "images should have rgb channels!"
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
        imgs = (imgs * std) + mean
        ret = imgs * 255
    else:
        raise ValueError()
    return ret.clamp(0, 255).to(torch.uint8)


def patching(images: torch.Tensor, patch_size: int):
    # [B T C H W]
    return rearrange(
        images,
        "... c (nh ph) (nw pw) -> ... (nh nw) (ph pw c)",
        pw=patch_size,
        ph=patch_size,
    )


def unpatching(
    patches: torch.Tensor, grid_size: int, patch_size: int, channels: int = 3
):
    # [B T (nh nw) (ph pw c)]
    return rearrange(
        patches,
        "... (nh nw) (ph pw c) -> ... c (nh ph) (nw pw)",
        nh=grid_size,
        pw=patch_size,
        c=channels,
    )
