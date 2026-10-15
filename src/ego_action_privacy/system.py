"""End-to-end differentiable protection system."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def freeze_target(target: nn.Module) -> nn.Module:
    target.requires_grad_(False)
    target.eval()
    return target


class ActionPrivacySystem(nn.Module):
    def __init__(self, generator: nn.Module, target: nn.Module, text_features: torch.Tensor) -> None:
        super().__init__()
        if text_features.ndim != 2 or not text_features.is_floating_point():
            raise ValueError("text_features must be floating point [classes, embedding]")
        self.generator = generator
        self.target = freeze_target(target)
        self.register_buffer("text_features", F.normalize(text_features.float(), dim=-1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()
        return self

    def forward(self, clean_video: torch.Tensor) -> dict[str, torch.Tensor]:
        delta = self.generator(clean_video)
        protected = (clean_video + delta).clamp(0.0, 1.0)
        logits = self.target(protected, self.text_features)
        return {"logits": logits, "protected": protected, "delta": delta}
