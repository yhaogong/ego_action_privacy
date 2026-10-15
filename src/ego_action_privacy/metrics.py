"""Classification, distortion, and temporal metrics."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    maximum = min(max(topk), logits.shape[1])
    predictions = logits.topk(maximum, dim=1).indices
    return {k: int(predictions[:, : min(k, maximum)].eq(labels[:, None]).any(dim=1).sum()) for k in topk}


def batch_psnr(clean: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    mse = (clean - protected).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def batch_ssim(clean: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    """Channel-wise global SSIM averaged over frames and channels."""
    if clean.shape != protected.shape or clean.ndim != 5:
        raise ValueError("Expected matching [B,T,C,H,W] videos")
    dimensions = (-2, -1)
    mean_x = clean.mean(dim=dimensions)
    mean_y = protected.mean(dim=dimensions)
    variance_x = clean.var(dim=dimensions, unbiased=False)
    variance_y = protected.var(dim=dimensions, unbiased=False)
    covariance = ((clean - mean_x[..., None, None]) * (protected - mean_y[..., None, None])).mean(dim=dimensions)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return ssim.mean(dim=(1, 2))


def distortion_metrics(clean: torch.Tensor, protected: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    if clean.shape != protected.shape or clean.shape != delta.shape:
        raise ValueError("clean, protected and delta must have matching shapes")
    temporal = delta[:, 1:] - delta[:, :-1] if delta.shape[1] > 1 else delta[:, :0]
    return {
        "linf": float(delta.detach().abs().amax()),
        "rmse": float(delta.detach().square().mean().sqrt()),
        "mae": float(delta.detach().abs().mean()),
        "temporal_delta_mae": float(temporal.detach().abs().mean()) if temporal.numel() else 0.0,
        "psnr": float(batch_psnr(clean.detach(), protected.detach()).mean()),
        "ssim": float(batch_ssim(clean.detach(), protected.detach()).mean()),
    }


def uniform_random_delta(video: torch.Tensor, epsilon: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.empty_like(video).uniform_(-epsilon, epsilon, generator=generator)
