from einops import rearrange
from torch import nn


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def mark_no_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


def flatten(x):
    return rearrange(x, "b t ... -> (b t) ...")


def unflatten(x, t):
    return rearrange(x, "(b t) ... -> b t ...", t=t)


def chw2hwc(x):
    return rearrange(x, "... c h w -> ... h w c")


def hwc2chw(x):
    return rearrange(x, "... h w c -> ... c h w")
