#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ego_action_privacy.config import load_config, set_seed, torch_dtype, write_json
from ego_action_privacy.data import EgoKeystepDataset, collate_samples
from ego_action_privacy.metrics import batch_psnr, batch_ssim, uniform_random_delta
from ego_action_privacy.system import freeze_target
from ego_action_privacy.targets import InternVideo2CLIP, InternVideo2Classifier
from ego_action_privacy.targets.text_cache import load_text_cache, read_class_map
from ego_action_privacy.training import load_generator_checkpoint


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Metrics:
    def __init__(self) -> None:
        self.samples = 0
        self.top1 = 0
        self.top5 = 0
        self.true_rank = 0.0
        self.psnr = 0.0
        self.ssim = 0.0
        self.linf = 0.0
        self.mae = 0.0
        self.temporal = 0.0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, clean=None, protected=None) -> None:
        batch = labels.numel()
        order = logits.detach().argsort(dim=1, descending=True)
        ranks = (order == labels[:, None]).long().argmax(1) + 1
        self.samples += batch
        self.top1 += int((ranks <= 1).sum())
        self.top5 += int((ranks <= min(5, logits.shape[1])).sum())
        self.true_rank += float(ranks.sum())
        if clean is not None and protected is not None:
            actual = protected - clean
            self.psnr += float(batch_psnr(clean, protected).sum())
            self.ssim += float(batch_ssim(clean, protected).sum())
            self.linf = max(self.linf, float(actual.abs().amax()))
            self.mae += float(actual.abs().flatten(1).mean(1).sum())
            temporal = actual[:, 1:] - actual[:, :-1]
            self.temporal += float(temporal.abs().flatten(1).mean(1).sum()) if temporal.numel() else 0.0

    def result(self) -> dict[str, float | int]:
        denominator = max(self.samples, 1)
        result: dict[str, float | int] = {
            "num_samples": self.samples,
            "top1_accuracy": self.top1 / denominator,
            "top5_accuracy": self.top5 / denominator,
            "mean_true_class_rank": self.true_rank / denominator,
        }
        if self.psnr:
            result.update({
                "psnr": self.psnr / denominator,
                "ssim": self.ssim / denominator,
                "linf": self.linf,
                "mae": self.mae / denominator,
                "temporal_delta_mae": self.temporal / denominator,
            })
        return result


def save_comparison(path: Path, clean: torch.Tensor, protected: torch.Tensor, epsilon: float, fps: float = 4.0) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Video export requires opencv-python-headless") from error
    clean = clean.detach().cpu().clamp(0, 1)
    protected = protected.detach().cpu().clamp(0, 1)
    difference = ((protected - clean) / (2 * epsilon) + 0.5).clamp(0, 1)
    frames = torch.cat((clean, protected, difference), dim=-1).mul(255).byte().permute(0, 2, 3, 1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean, protected, and equal-budget random-noise videos")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Omit for clean baseline only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial-data", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    set_seed(int(config["evaluation"]["seed"]))
    allow_partial = args.allow_partial_data or bool(config["data"].get("allow_partial_data", False))
    dataset = EgoKeystepDataset(
        config["paths"]["val_manifest"],
        num_frames=int(config["data"]["num_frames"]),
        output_size=int(config["data"]["image_size"]),
        availability="available" if allow_partial else "valid",
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_samples,
        pin_memory=args.device.startswith("cuda"),
    )
    target_kind = config["target"].get("kind", "zero_shot")
    text = None
    if target_kind == "supervised":
        raw_classes, _ = read_class_map(config["paths"]["class_map"])
        target = InternVideo2Classifier(
            config["paths"]["model_dir"],
            len(raw_classes),
            device=args.device,
            parameter_dtype=torch_dtype(config["target"]["dtype"]),
            gradient_checkpointing=False,
            head_dropout=0.0,
        )
        target.load_finetuned_checkpoint(
            config["paths"]["recognizer_checkpoint"],
            class_map_path=config["paths"]["class_map"],
        )
    elif target_kind == "zero_shot":
        text = load_text_cache(
            config["paths"]["text_cache"], config["paths"]["class_map"]
        ).to(args.device)
        target = InternVideo2CLIP(
            config["paths"]["model_dir"],
            device=args.device,
            dtype=torch_dtype(config["target"]["dtype"]),
            gradient_checkpointing=False,
            load_text_encoder=False,
        )
    else:
        raise ValueError(f"Unsupported target kind: {target_kind!r}")
    freeze_target(target)
    generator = None
    epsilon = float(config["generator"]["epsilon_255"]) / 255.0
    if args.checkpoint:
        generator, checkpoint = load_generator_checkpoint(args.checkpoint, args.device)
        generator.eval()
        epsilon = generator.epsilon
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_classes, names = read_class_map(config["paths"]["class_map"])
    class_map = json.loads(Path(config["paths"]["class_map"]).read_text(encoding="utf-8"))["classes"]
    unseen_indices = {row["class_index"] for row in class_map if row["train_count"] == 0 and row["val_count"] > 0}
    metrics = {"clean": Metrics()}
    if generator is not None:
        metrics.update(protected=Metrics(), random=Metrics())
    unseen_metrics = {name: Metrics() for name in metrics}
    class_stats = {
        name: defaultdict(lambda: {"support": 0, "top1_correct": 0, "top5_correct": 0})
        for name in metrics
    }
    predictions = (output / "predictions.jsonl").open("w", encoding="utf-8")
    random_generator = torch.Generator(device=args.device).manual_seed(int(config["evaluation"]["random_noise_seed"]))
    example_count = 0
    processed = 0
    try:
        for batch in loader:
            clean = batch["video"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            with torch.no_grad():
                condition_logits = {
                    "clean": target(clean) if text is None else target(clean, text)
                }
                condition_videos = {"clean": clean}
                if generator is not None:
                    delta = generator(clean)
                    condition_videos["protected"] = (clean + delta).clamp(0, 1)
                    random_delta = uniform_random_delta(clean, epsilon, generator=random_generator)
                    condition_videos["random"] = (clean + random_delta).clamp(0, 1)
                    condition_logits["protected"] = (
                        target(condition_videos["protected"])
                        if text is None else target(condition_videos["protected"], text)
                    )
                    condition_logits["random"] = (
                        target(condition_videos["random"])
                        if text is None else target(condition_videos["random"], text)
                    )
            unseen_mask = torch.tensor([int(label) in unseen_indices for label in labels], device=labels.device)
            for name, logits in condition_logits.items():
                distortion_video = None if name == "clean" else condition_videos[name]
                metrics[name].update(logits, labels, clean if distortion_video is not None else None, distortion_video)
                if unseen_mask.any():
                    unseen_metrics[name].update(
                        logits[unseen_mask],
                        labels[unseen_mask],
                        clean[unseen_mask] if distortion_video is not None else None,
                        distortion_video[unseen_mask] if distortion_video is not None else None,
                    )
                order = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
                for row_index, label in enumerate(labels.tolist()):
                    stats = class_stats[name][label]
                    stats["support"] += 1
                    stats["top1_correct"] += int(order[row_index, 0].item() == label)
                    stats["top5_correct"] += int(label in order[row_index].tolist())
            for index, metadata in enumerate(batch["metadata"]):
                record = {"sample_id": metadata["sample_id"], "label": int(labels[index]), "raw_class": metadata["raw_class"]}
                for name, logits in condition_logits.items():
                    top = logits[index].topk(5)
                    record[name] = {
                        "top5_indices": top.indices.cpu().tolist(),
                        "top5_names": [names[item] for item in top.indices.cpu().tolist()],
                        "top5_logits": top.values.float().cpu().tolist(),
                    }
                predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                if generator is not None and example_count < int(config["evaluation"]["example_videos"]):
                    save_comparison(
                        output / "examples" / f"{metadata['sample_id']}_clean_protected_delta.mp4",
                        clean[index],
                        condition_videos["protected"][index],
                        epsilon,
                    )
                    example_count += 1
            processed += labels.numel()
            if processed == len(dataset) or processed % 50 == 0:
                progress = {
                    name: value.top1 / max(value.samples, 1)
                    for name, value in metrics.items()
                }
                print(f"[{processed}/{len(dataset)}] top1={progress}", flush=True)
    finally:
        predictions.close()
    with (output / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "condition", "class_index", "raw_class", "action_name", "train_annotation_count",
            "val_annotation_count", "train_usable_count", "val_usable_count", "selected_support", "top1_correct", "top5_correct",
            "top1_accuracy", "top5_accuracy",
        ])
        writer.writeheader()
        for condition, condition_stats in class_stats.items():
            for row in class_map:
                stats = condition_stats[row["class_index"]]
                support = stats["support"]
                writer.writerow({
                    "condition": condition,
                    "class_index": row["class_index"],
                    "raw_class": row["raw_class"],
                    "action_name": row["action_name"],
                    "train_annotation_count": row["train_count"],
                    "val_annotation_count": row["val_count"],
                    "train_usable_count": row.get("train_usable_count", ""),
                    "val_usable_count": row.get("val_usable_count", ""),
                    "selected_support": support,
                    "top1_correct": stats["top1_correct"],
                    "top5_correct": stats["top5_correct"],
                    "top1_accuracy": stats["top1_correct"] / support if support else "",
                    "top5_accuracy": stats["top5_correct"] / support if support else "",
                })
    report = {
        "complete": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation split; current file coverage reported below",
        "partial_data": allow_partial,
        "dataset_coverage": dataset.coverage,
        "num_classes": len(raw_classes),
        "val_classes_absent_from_train": len(unseen_indices),
        "epsilon": epsilon,
        "epsilon_255": epsilon * 255,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "protocol": {
            "manifest": str(Path(config["paths"]["val_manifest"]).resolve()),
            "manifest_sha256": sha256(config["paths"]["val_manifest"]),
            "class_map": str(Path(config["paths"]["class_map"]).resolve()),
            "class_map_sha256": sha256(config["paths"]["class_map"]),
            "target_kind": target_kind,
            "text_cache": (
                str(Path(config["paths"]["text_cache"]).resolve())
                if text is not None else None
            ),
            "recognizer_checkpoint": (
                str(Path(config["paths"]["recognizer_checkpoint"]).resolve())
                if target_kind == "supervised" else None
            ),
            "target": config["target"]["name"],
            "source_commit": "3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3",
            "num_frames": int(config["data"]["num_frames"]),
            "generator_input_size": int(config["data"]["image_size"]),
            "sampling": "uniform rounded indices inside annotated [start_time,end_time)",
            "target_resize": "224x224 bicubic antialiased",
            "target_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "text_prefix": "summarize:" if text is not None else None,
            "text_max_tokens": 80 if text is not None else None,
            "torch": torch.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        },
        "metrics": {name: value.result() for name, value in metrics.items()},
        "unseen_class_metrics": {name: value.result() for name, value in unseen_metrics.items()},
        "random_noise": "independent uniform U[-epsilon,epsilon], clipped to [0,1]",
        "example_layout": "left=clean, center=protected, right=delta mapped from [-epsilon,epsilon] to [0,1]",
        "load_report": target.load_report,
    }
    write_json(output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
