from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from typing_extensions import Literal


def images2gif(images, filename: str = "temp.gif"):
    images = [topil(img) for img in images]
    # fps = 1
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=100,
        loop=0,
    )


def save_images(
    images,
    fp: str = "temp",
    output_format: Literal["gif", "mp4", "png"] | None = None,
    fps: float = 5.0,
):
    if output_format is None:
        output_format = Path(fp).suffix[1:]
        fp = Path(fp).parent / Path(fp).stem
    else:
        fp = fp
    if output_format == "gif":
        images2gif(images, f"{fp}.gif")
    elif output_format == "mp4":
        from torchvision.io import write_video

        x = totensor(images)
        write_video(f"{fp}.mp4", x, fps=fps, video_codec="h264")
    elif output_format == "png":
        import os

        os.makedirs(fp, exist_ok=True)
        for i, img in enumerate(images):
            topil(img).save(f"{fp}/{i}.png")


def totensor(img: np.ndarray | torch.Tensor | Image.Image):
    if isinstance(img, torch.Tensor):
        x = img.cpu().detach()
    elif isinstance(img, Image.Image):
        x = torch.tensor(np.array(img))
    else:
        if isinstance(img, list):
            img = np.array(img)
        x = torch.tensor(img)

    if x.shape[-1] != 3:
        x = rearrange(x, "... c h w -> ... h w c")
    return x


def topil(img: np.ndarray | torch.Tensor | Image.Image):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    if isinstance(img, Image.Image):
        return img
    img = img.astype(np.uint8)

    if img.shape[-1] != 3:
        img = rearrange(img, "... c h w -> ... h w c")

    img = Image.fromarray(img)
    return img
