"""Framewise and temporally coupled U-Net perturbation generators."""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return max(group for group in range(1, min(8, channels) + 1) if channels % group == 0)


class _ConvBlock2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _ConvBlock3d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _UNet2d(nn.Module):
    def __init__(self, channels: int, base_channels: int) -> None:
        super().__init__()
        self.encoder0 = _ConvBlock2d(channels, base_channels)
        self.encoder1 = _ConvBlock2d(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock2d(base_channels * 2, base_channels * 4)
        self.decoder1 = _ConvBlock2d(base_channels * 6, base_channels * 2)
        self.decoder0 = _ConvBlock2d(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder0(value)
        skip1 = self.encoder1(self.pool(skip0))
        value = self.bottleneck(self.pool(skip1))
        value = F.interpolate(value, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decoder1(torch.cat((value, skip1), dim=1))
        value = F.interpolate(value, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder0(torch.cat((value, skip0), dim=1)))


class _UNet3d(nn.Module):
    def __init__(self, channels: int, base_channels: int) -> None:
        super().__init__()
        self.encoder0 = _ConvBlock3d(channels, base_channels)
        self.encoder1 = _ConvBlock3d(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock3d(base_channels * 2, base_channels * 4)
        self.decoder1 = _ConvBlock3d(base_channels * 6, base_channels * 2)
        self.decoder0 = _ConvBlock3d(base_channels * 3, base_channels)
        self.output = nn.Conv3d(base_channels, channels, 1)
        # Preserve annotated temporal sampling; downsample only spatial dimensions.
        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder0(value)
        skip1 = self.encoder1(self.pool(skip0))
        value = self.bottleneck(self.pool(skip1))
        value = F.interpolate(value, size=skip1.shape[-3:], mode="trilinear", align_corners=False)
        value = self.decoder1(torch.cat((value, skip1), dim=1))
        value = F.interpolate(value, size=skip0.shape[-3:], mode="trilinear", align_corners=False)
        return self.output(self.decoder0(torch.cat((value, skip0), dim=1)))


class VideoPerturbationUNet(nn.Module):
    """Generate bounded RGB perturbations for `[B,T,3,H,W]` input.

    `temporal_mode="3d"` mixes neighboring frames. `"framewise"` applies the
    exact same 2-D U-Net independently to each frame and is the temporal-module
    ablation baseline.
    """

    def __init__(
        self,
        *,
        epsilon: float = 8 / 255,
        base_channels: int = 32,
        temporal_mode: Literal["3d", "framewise"] = "3d",
    ) -> None:
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0 or epsilon > 1:
            raise ValueError("epsilon must lie in (0, 1]")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if temporal_mode not in ("3d", "framewise"):
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")
        self.epsilon = float(epsilon)
        self.temporal_mode = temporal_mode
        self.network = _UNet3d(3, base_channels) if temporal_mode == "3d" else _UNet2d(3, base_channels)
        # Start from an identity protection mapping while keeping useful hidden activations.
        nn.init.zeros_(self.network.output.weight)
        nn.init.zeros_(self.network.output.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Expected [B,T,3,H,W], received {tuple(video.shape)}")
        batch, frames, channels, height, width = video.shape
        if self.temporal_mode == "framewise":
            raw = self.network(video.reshape(batch * frames, channels, height, width))
            raw = raw.reshape(batch, frames, channels, height, width)
        else:
            raw = self.network(video.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return self.epsilon * torch.tanh(raw)

    def protect(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self(video)
        return (video + delta).clamp(0.0, 1.0), delta
