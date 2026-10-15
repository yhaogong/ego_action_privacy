"""Validated serialization of fixed class text features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def read_class_map(path: str | Path) -> tuple[list[str], list[str]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = value.get("classes") if isinstance(value, dict) else None
    if not isinstance(classes, list) or value.get("num_classes") != len(classes):
        raise ValueError("Invalid class_map.json")
    ordered = sorted(classes, key=lambda row: row["class_index"])
    if [row["class_index"] for row in ordered] != list(range(len(ordered))):
        raise ValueError("class_index must be contiguous from zero")
    raw_classes = [str(row["raw_class"]) for row in ordered]
    names = [row["action_name"] for row in ordered]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Empty class name")
    return raw_classes, names


def class_map_sha256(path: str | Path) -> str:
    """Hash only index/raw-id/text semantics, not mutable coverage counts."""
    raw_classes, names = read_class_map(path)
    canonical = json.dumps(
        [{"class_index": index, "raw_class": raw, "action_name": name}
         for index, (raw, name) in enumerate(zip(raw_classes, names))],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_text_cache(
    path: str | Path,
    features: torch.Tensor,
    class_map: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    raw_classes, names = read_class_map(class_map)
    if features.shape[0] != len(names) or features.ndim != 2:
        raise ValueError("Text feature count does not match class map")
    if not torch.isfinite(features).all():
        raise ValueError("Text features contain NaN/Inf")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "features": F.normalize(features.detach().float().cpu(), dim=-1),
        "raw_classes": raw_classes,
        "class_names": names,
        "class_map_sha256": class_map_sha256(class_map),
        "metadata": metadata or {},
    }, destination)


def load_text_cache(path: str | Path, class_map: str | Path) -> torch.Tensor:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported text feature cache")
    raw_classes, names = read_class_map(class_map)
    if value.get("class_map_sha256") != class_map_sha256(class_map):
        raise ValueError("Text cache was built for a different class_map.json")
    if value.get("raw_classes") != raw_classes or value.get("class_names") != names:
        raise ValueError("Text cache class order differs from class map")
    features = value.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != len(names):
        raise ValueError("Invalid cached feature tensor")
    if not torch.isfinite(features).all():
        raise ValueError("Cached text features contain NaN/Inf")
    return F.normalize(features.float(), dim=-1)
