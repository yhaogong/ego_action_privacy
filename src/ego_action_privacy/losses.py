"""Optimization objectives for non-targeted action privacy."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def untargeted_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Minimize true-class logit below the strongest alternative by `margin`."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected logits [B,C] and labels [B]")
    if logits.shape[1] < 2:
        raise ValueError("Untargeted classification requires at least two classes")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), float("-inf"))
    other_logits = masked.amax(dim=1)
    return F.relu(true_logits - other_logits + margin).mean()


def actual_pixel_loss(clean_video: torch.Tensor, protected_video: torch.Tensor) -> torch.Tensor:
    """Measure the realized pixel MSE after the protected video is clamped."""
    if clean_video.ndim != 5 or clean_video.shape != protected_video.shape:
        raise ValueError("Expected matching clean and protected videos [B,T,C,H,W]")
    return F.mse_loss(protected_video, clean_video)


def temporal_difference_loss(delta: torch.Tensor, p: int = 1) -> torch.Tensor:
    if delta.ndim != 5:
        raise ValueError("Expected delta [B,T,C,H,W]")
    if delta.shape[1] < 2:
        return delta.new_zeros(())
    difference = delta[:, 1:] - delta[:, :-1]
    return difference.abs().mean() if p == 1 else difference.square().mean()


def spatial_total_variation(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 5:
        raise ValueError("Expected delta [B,T,C,H,W]")
    horizontal = (delta[..., 1:] - delta[..., :-1]).abs().mean()
    vertical = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
    return horizontal + vertical


@dataclass(frozen=True)
class LossWeights:
    attack: float = 1.0
    pixel: float = 0.05
    magnitude: float = 0.05
    temporal: float = 0.1
    spatial_tv: float = 0.0
    margin: float = 0.0


def protection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clean_video: torch.Tensor,
    protected_video: torch.Tensor,
    delta: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    attack = untargeted_margin_loss(logits, labels, weights.margin)
    pixel = actual_pixel_loss(clean_video, protected_video)
    magnitude = delta.square().mean()
    temporal = temporal_difference_loss(delta)
    spatial = spatial_total_variation(delta)
    total = (
        weights.attack * attack
        + weights.pixel * pixel
        + weights.magnitude * magnitude
        + weights.temporal * temporal
        + weights.spatial_tv * spatial
    )
    return {
        "total": total,
        "attack": attack,
        "pixel": pixel,
        "magnitude": magnitude,
        "temporal": temporal,
        "spatial_tv": spatial,
    }
