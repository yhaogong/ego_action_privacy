#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ego_action_privacy.targets import InternVideo2CLIP
from ego_action_privacy.targets.text_cache import read_class_map, save_text_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache all Keystep class text features once")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    raw_classes, names = read_class_map(args.class_map)
    model = InternVideo2CLIP(
        args.model_dir,
        device=args.device,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        gradient_checkpointing=False,
    )
    features = model.encode_text(names, batch_size=args.batch_size)
    save_text_cache(
        args.output,
        features,
        args.class_map,
        metadata={
            "model": "InternVideo2-CLIP-1B-224p-f8",
            "text_prefix": "summarize:",
            "max_tokens": 80,
            "num_classes": len(names),
            "embedding_dim": features.shape[1],
            "load_report": model.load_report,
        },
    )
    report = {
        "output": str(Path(args.output).resolve()),
        "num_classes": len(raw_classes),
        "shape": list(features.shape),
        "norm_min": float(features.norm(dim=-1).min()),
        "norm_max": float(features.norm(dim=-1).max()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

