#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ego_action_privacy.targets.text_cache import load_text_cache, read_class_map, save_text_cache


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an ordered class subset from an existing text cache")
    parser.add_argument("--source-class-map", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--target-class-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_raw, source_names = read_class_map(args.source_class_map)
    target_raw, target_names = read_class_map(args.target_class_map)
    source_features = load_text_cache(args.source_cache, args.source_class_map)
    source_indices = {raw: index for index, raw in enumerate(source_raw)}
    selected = []
    for raw, name in zip(target_raw, target_names):
        if raw not in source_indices:
            raise ValueError(f"Target raw class {raw!r} is absent from source cache")
        index = source_indices[raw]
        if source_names[index] != name:
            raise ValueError(f"Class name mismatch for raw class {raw!r}")
        selected.append(source_features[index])
    features = torch.stack(selected)
    save_text_cache(
        args.output,
        features,
        args.target_class_map,
        metadata={
            "model": "InternVideo2-CLIP-1B-224p-f8",
            "selection": "ordered exact step_unique_id/name subset of validated full-vocabulary cache",
            "source_cache": str(Path(args.source_cache).resolve()),
            "source_cache_sha256": sha256(args.source_cache),
            "source_class_map": str(Path(args.source_class_map).resolve()),
            "source_class_map_sha256": sha256(args.source_class_map),
            "num_classes": len(target_raw),
            "embedding_dim": features.shape[1],
        },
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "num_classes": len(target_raw),
        "shape": list(features.shape),
        "norm_min": float(features.norm(dim=-1).min()),
        "norm_max": float(features.norm(dim=-1).max()),
    }, indent=2))


if __name__ == "__main__":
    main()
