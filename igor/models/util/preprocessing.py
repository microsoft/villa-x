import torch


def normalize_images(img, img_norm_type="imagenet"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.to(torch.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.to(torch.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3)).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3)).to(img.device)

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = torch.tile(mean, num_tile)
        std_tile = torch.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    raise ValueError()


def denormalize_images(img: torch.Tensor, img_norm_type="imagenet"):
    if img_norm_type == "default":
        # put pixels in [0, 255]
        ret = (img + 1.0) * 127.5
    elif img_norm_type == "imagenet":
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3)).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3)).to(img.device)

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = torch.tile(mean, num_tile)
        std_tile = torch.tile(std, num_tile)

        ret = (img * std_tile + mean_tile) * 255
    else:
        raise ValueError()
    return ret.clamp(0, 255).to(torch.uint8)
