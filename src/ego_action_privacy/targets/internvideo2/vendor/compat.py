"""Small PyTorch-only replacements for timm utility imports."""
from collections.abc import Iterable
import torch
from torch import nn
from torch.nn.init import trunc_normal_

def to_2tuple(value):
    return tuple(value) if isinstance(value, Iterable) else (value, value)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
