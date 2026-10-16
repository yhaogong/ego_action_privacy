#!/usr/bin/env python3
"""Real-video target forward and RGB-gradient check."""
from __future__ import annotations

import argparse
import json
import time

import torch

from ego_action_privacy.config import write_json
from ego_action_privacy.data import EgoKeystepDataset
from ego_action_privacy.targets import InternVideo2CLIP
from ego_action_privacy.targets.text_cache import load_text_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dataset = EgoKeystepDataset(args.manifest, availability="available", limit=args.sample_index + 1)
    sample = dataset[args.sample_index]
    video = sample["video"].unsqueeze(0).to(args.device).requires_grad_(True)
    label = int(sample["label"])
    text = load_text_cache(args.text_cache, args.class_map).to(args.device)
    started = time.perf_counter()
    model = InternVideo2CLIP(
        args.model_dir,
        device=args.device,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        gradient_checkpointing=True,
        load_text_encoder=False,
    )
    logits = model(video, text)
    logits[0, label].backward()
    if video.grad is None or not torch.isfinite(video.grad).all() or video.grad.abs().max() == 0:
        raise RuntimeError("RGB gradient is missing, zero, or nonfinite")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen target accumulated parameter gradients")
    top = logits.detach().softmax(-1)[0].topk(5)
    report = {
        "passed": True,
        "sample": sample["metadata"],
        "sampled_frame_indices": sample["decode"]["indices"],
        "logits_shape": list(logits.shape),
        "top5_class_indices": top.indices.cpu().tolist(),
        "top5_probabilities": top.values.cpu().tolist(),
        "input_gradient_abs_max": float(video.grad.abs().max()),
        "input_gradient_abs_mean": float(video.grad.abs().mean()),
        "frozen_parameter_grads": False,
        "load_report": model.load_report,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("TARGET SMOKE PASSED")


if __name__ == "__main__":
    main()

