"""Training/checkpoint helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .models import VideoPerturbationUNet


def generator_from_config(config: dict[str, Any]) -> VideoPerturbationUNet:
    return VideoPerturbationUNet(
        epsilon=float(config["epsilon_255"]) / 255.0,
        base_channels=int(config["base_channels"]),
        temporal_mode=config["temporal_mode"],
    )


def save_checkpoint(
    path: str | Path,
    generator: VideoPerturbationUNet,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    epoch: int,
    generator_config: dict[str, Any],
    full_config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "schema_version": 1,
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "generator_config": generator_config,
        "config": full_config,
    }, temporary)
    temporary.replace(destination)


def load_generator_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[VideoPerturbationUNet, dict[str, Any]]:
    value = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported generator checkpoint")
    config = value.get("generator_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no generator configuration")
    model = generator_from_config(config).to(device)
    model.load_state_dict(value["generator"], strict=True)
    return model, value


def trainable_parameter_checksum(module: torch.nn.Module) -> float:
    return float(sum(parameter.detach().float().sum() for parameter in module.parameters() if parameter.requires_grad))
