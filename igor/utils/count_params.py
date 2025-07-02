from typing import Type

import torch.nn as nn


def count_parameters(model: Type[nn.Module]) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
