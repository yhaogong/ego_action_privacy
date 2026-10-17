#!/usr/bin/env python3
"""Estimate the per-video white-box attack upper bound under an L-inf budget.

This script does not train a generator.  It directly optimizes one perturbation
per clean-correct validation clip against the frozen supervised recognizer.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ego_action_privacy.config import load_config, set_seed, torch_dtype, write_json
from ego_action_privacy.data import EgoKeystepDataset, collate_samples
from ego_action_privacy.metrics import batch_psnr, batch_ssim
from ego_action_privacy.system import freeze_target
from ego_action_privacy.targets import InternVideo2Classifier
from ego_action_privacy.targets.text_cache import read_class_map


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_record_steps(value: str, max_steps: int) -> list[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps or steps[0] <= 0:
        raise ValueError("--record-steps must contain positive integers")
    if steps[-1] > max_steps:
        raise ValueError("--record-steps cannot exceed --steps")
    if max_steps not in steps:
        steps.append(max_steps)
    return steps


def read_clean_correct(path: str | Path) -> dict[str, int]:
    """Return sample_id -> label for samples classified correctly at clean top-1."""
    result: dict[str, int] = {}
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            label = row.get("label")
            top5 = row.get("clean", {}).get("top5_indices")
            if not isinstance(sample_id, str) or not isinstance(label, int) or not isinstance(top5, list):
                raise ValueError(f"Malformed clean prediction at line {line_number}: {source}")
            if sample_id in result:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {source}")
            if top5 and int(top5[0]) == label:
                result[sample_id] = label
    if not result:
        raise ValueError(f"No clean-correct samples found in {source}")
    return result


def select_clean_correct_indices(
    dataset: EgoKeystepDataset,
    clean_correct: dict[str, int],
    *,
    limit: int | None,
    seed: int,
) -> list[int]:
    """Select a deterministic, approximately class-balanced calibration subset."""
    grouped: dict[int, list[int]] = defaultdict(list)
    matched_ids: set[str] = set()
    for index in range(len(dataset)):
        metadata = dataset.metadata(index)
        sample_id = metadata["sample_id"]
        if sample_id not in clean_correct:
            continue
        label = int(metadata["label"])
        if clean_correct[sample_id] != label:
            raise ValueError(f"Label mismatch for clean-correct sample {sample_id}")
        grouped[label].append(index)
        matched_ids.add(sample_id)
    missing = sorted(set(clean_correct) - matched_ids)
    if missing:
        raise ValueError(
            f"Clean predictions contain {len(missing)} samples absent from the selected dataset; "
            f"first missing sample: {missing[0]}"
        )
    total = sum(len(values) for values in grouped.values())
    if limit is None or limit >= total:
        return [index for values in grouped.values() for index in values]
    if limit <= 0:
        raise ValueError("--limit-correct must be positive")

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    labels = sorted(grouped)
    rng.shuffle(labels)
    selected: list[int] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for label in labels:
            values = grouped[label]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def attack_score(logits: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, labels, reduction="none")
    if loss_name == "margin":
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, labels[:, None], float("-inf"))
        return masked.amax(dim=1) - true_logits
    raise ValueError(f"Unsupported attack loss: {loss_name}")


def project_linf(candidate: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = (candidate - clean).clamp(-epsilon, epsilon)
    return (clean + delta).clamp(0.0, 1.0)


def pgd_snapshots(
    target: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    record_steps: list[int],
    restarts: int,
    random_start: bool,
    loss_name: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor | float]]]:
    """Return clean logits and best-by-objective candidates at requested steps."""
    if clean.shape[0] != 1 or labels.numel() != 1:
        raise ValueError("The upper-bound attack currently requires batch size 1")
    with torch.no_grad():
        clean_logits = target(clean).float()
        clean_score = float(attack_score(clean_logits, labels, loss_name).item())

    overall: dict[int, dict[str, torch.Tensor | float]] = {}
    wanted = set(record_steps)
    for _ in range(restarts):
        if random_start:
            delta = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
            current = project_linf(clean + delta, clean, epsilon)
        else:
            current = clean.clone()
        best_adv = clean.detach().clone()
        best_logits = clean_logits.detach().clone()
        best_score = clean_score

        for step in range(steps + 1):
            current = current.detach().requires_grad_(step < steps)
            logits = target(current).float()
            scores = attack_score(logits, labels, loss_name)
            score = float(scores.item())
            if score > best_score:
                best_score = score
                best_adv = current.detach().clone()
                best_logits = logits.detach().clone()
            if step in wanted:
                previous = overall.get(step)
                if previous is None or best_score > float(previous["score"]):
                    overall[step] = {
                        "score": best_score,
                        "video": best_adv.detach().clone(),
                        "logits": best_logits.detach().clone(),
                    }
            if step == steps:
                break
            gradient = torch.autograd.grad(scores.sum(), current, only_inputs=True)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"Nonfinite input gradient at PGD step {step}")
            current = project_linf(current + step_size * gradient.sign(), clean, epsilon).detach()
    missing = wanted - set(overall)
    if missing:
        raise RuntimeError(f"Missing PGD snapshots: {sorted(missing)}")
    return clean_logits, overall


def candidate_result(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    objective: float,
) -> dict[str, Any]:
    order = logits.argsort(dim=1, descending=True)
    rank = int(((order == labels[:, None]).long().argmax(1) + 1).item())
    top = logits.topk(min(5, logits.shape[1]), dim=1)
    delta = adversarial - clean
    temporal = delta[:, 1:] - delta[:, :-1]
    return {
        "objective": objective,
        "true_class_rank": rank,
        "attack_success": rank > 1,
        "top5_evicted": rank > 5,
        "top5_indices": top.indices[0].cpu().tolist(),
        "top5_logits": top.values[0].float().cpu().tolist(),
        "linf": float(delta.abs().amax()),
        "mae": float(delta.abs().flatten(1).mean(1).item()),
        "temporal_delta_mae": float(temporal.abs().flatten(1).mean(1).item()) if temporal.numel() else 0.0,
        "psnr": float(batch_psnr(clean, adversarial).item()),
        "ssim": float(batch_ssim(clean, adversarial).item()),
    }


class Aggregate:
    def __init__(self) -> None:
        self.samples = 0
        self.successes = 0
        self.top5_evictions = 0
        self.rank_sum = 0.0
        self.objective_sum = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.mae_sum = 0.0
        self.temporal_sum = 0.0
        self.max_linf = 0.0

    def update(self, result: dict[str, Any]) -> None:
        self.samples += 1
        self.successes += int(result["attack_success"])
        self.top5_evictions += int(result["top5_evicted"])
        self.rank_sum += float(result["true_class_rank"])
        self.objective_sum += float(result["objective"])
        self.psnr_sum += float(result["psnr"])
        self.ssim_sum += float(result["ssim"])
        self.mae_sum += float(result["mae"])
        self.temporal_sum += float(result["temporal_delta_mae"])
        self.max_linf = max(self.max_linf, float(result["linf"]))

    def result(self) -> dict[str, int | float]:
        denominator = max(self.samples, 1)
        return {
            "num_clean_correct_attacked": self.samples,
            "attack_success_count": self.successes,
            "conditional_attack_success_rate": self.successes / denominator,
            "top5_eviction_count": self.top5_evictions,
            "conditional_top5_eviction_rate": self.top5_evictions / denominator,
            "mean_true_class_rank": self.rank_sum / denominator,
            "mean_objective": self.objective_sum / denominator,
            "psnr": self.psnr_sum / denominator,
            "ssim": self.ssim_sum / denominator,
            "linf": self.max_linf,
            "mae": self.mae_sum / denominator,
            "temporal_delta_mae": self.temporal_sum / denominator,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a per-video white-box PGD upper bound on clean-correct validation clips"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clean-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial-data", action="store_true")
    parser.add_argument("--limit-correct", type=int)
    parser.add_argument("--selection-seed", type=int, default=20261017)
    parser.add_argument("--attack-seed", type=int, default=20261018)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--record-steps", default="1,5,10,20,50,100")
    parser.add_argument("--step-size-255", type=float, default=1.0)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--loss", choices=("ce", "margin"), default="ce")
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.steps <= 0 or args.restarts <= 0 or args.step_size_255 <= 0:
        raise ValueError("steps, restarts, and step-size must be positive")
    if args.no_random_start and args.restarts > 1:
        raise ValueError("Multiple identical restarts require random starts")
    record_steps = parse_record_steps(args.record_steps, args.steps)
    config = load_config(args.config)
    if config["target"].get("kind") != "supervised":
        raise ValueError("Upper-bound evaluation currently requires target.kind=supervised")
    set_seed(args.attack_seed)

    allow_partial = args.allow_partial_data or bool(config["data"].get("allow_partial_data", False))
    dataset = EgoKeystepDataset(
        config["paths"]["val_manifest"],
        num_frames=int(config["data"]["num_frames"]),
        output_size=int(config["data"]["image_size"]),
        availability="available" if allow_partial else "valid",
    )
    clean_correct = read_clean_correct(args.clean_predictions)
    selected_indices = select_clean_correct_indices(
        dataset,
        clean_correct,
        limit=args.limit_correct,
        seed=args.selection_seed,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_samples,
        pin_memory=args.device.startswith("cuda"),
    )

    raw_classes, _ = read_class_map(config["paths"]["class_map"])
    target = InternVideo2Classifier(
        config["paths"]["model_dir"],
        len(raw_classes),
        device=args.device,
        parameter_dtype=torch_dtype(config["target"]["dtype"]),
        gradient_checkpointing=False,
        head_dropout=0.0,
    )
    checkpoint_report = target.load_finetuned_checkpoint(
        config["paths"]["recognizer_checkpoint"],
        class_map_path=config["paths"]["class_map"],
    )
    freeze_target(target)
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise RuntimeError("Target recognizer is not fully frozen")

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    if predictions_path.exists() and predictions_path.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite existing predictions: {predictions_path}")

    epsilon = float(config["generator"]["epsilon_255"]) / 255.0
    step_size = args.step_size_255 / 255.0
    selected_class_count = len({dataset.metadata(index)["label"] for index in selected_indices})
    common = {
        "schema_version": 1,
        "started_at": utc_now(),
        "status": "running",
        "protocol": "per_video_white_box_pgd_on_clean_correct_validation_samples",
        "config": str(Path(args.config).expanduser().resolve()),
        "clean_predictions": str(Path(args.clean_predictions).expanduser().resolve()),
        "recognizer_checkpoint": str(Path(config["paths"]["recognizer_checkpoint"]).expanduser().resolve()),
        "checkpoint_report": checkpoint_report,
        "dataset_coverage": dataset.coverage,
        "total_clean_correct": len(clean_correct),
        "selected_clean_correct": len(selected_indices),
        "selected_class_count": selected_class_count,
        "selection_seed": args.selection_seed,
        "attack_seed": args.attack_seed,
        "attack": {
            "loss": args.loss,
            "epsilon": epsilon,
            "epsilon_255": epsilon * 255.0,
            "step_size": step_size,
            "step_size_255": args.step_size_255,
            "steps": args.steps,
            "record_steps": record_steps,
            "restarts": args.restarts,
            "random_start": not args.no_random_start,
            "candidate_selection": "maximum attack objective seen so far",
        },
    }
    write_json(output / "status.json", common)
    aggregates = {step: Aggregate() for step in record_steps}
    attack_generator = torch.Generator(device=args.device).manual_seed(args.attack_seed)
    started = time.monotonic()

    try:
        with predictions_path.open("w", encoding="utf-8") as predictions:
            for sample_number, batch in enumerate(loader, start=1):
                clean = batch["video"].to(args.device, non_blocking=True)
                labels = batch["labels"].to(args.device, non_blocking=True)
                metadata = batch["metadata"][0]
                clean_logits, snapshots = pgd_snapshots(
                    target,
                    clean,
                    labels,
                    epsilon=epsilon,
                    step_size=step_size,
                    steps=args.steps,
                    record_steps=record_steps,
                    restarts=args.restarts,
                    random_start=not args.no_random_start,
                    loss_name=args.loss,
                    generator=attack_generator,
                )
                clean_rank = int(
                    ((clean_logits.argsort(dim=1, descending=True) == labels[:, None]).long().argmax(1) + 1).item()
                )
                if clean_rank != 1:
                    raise RuntimeError(
                        f"Stored clean prediction disagrees with current recognizer for {metadata['sample_id']}: "
                        f"rank={clean_rank}"
                    )
                record: dict[str, Any] = {
                    "sample_id": metadata["sample_id"],
                    "label": int(labels.item()),
                    "raw_class": metadata["raw_class"],
                    "clean_true_class_rank": clean_rank,
                    "steps": {},
                }
                for step in record_steps:
                    snapshot = snapshots[step]
                    result = candidate_result(
                        snapshot["logits"],
                        labels,
                        clean,
                        snapshot["video"],
                        float(snapshot["score"]),
                    )
                    if result["linf"] > epsilon + 1e-6:
                        raise RuntimeError(
                            f"L-inf budget violation for {metadata['sample_id']}: {result['linf']} > {epsilon}"
                        )
                    aggregates[step].update(result)
                    record["steps"][str(step)] = result
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()

                if sample_number % args.progress_every == 0 or sample_number == len(selected_indices):
                    elapsed = time.monotonic() - started
                    latest = aggregates[record_steps[-1]].result()
                    progress = {
                        **common,
                        "processed": sample_number,
                        "elapsed_seconds": elapsed,
                        "seconds_per_sample": elapsed / sample_number,
                        "estimated_remaining_seconds": (elapsed / sample_number) * (len(selected_indices) - sample_number),
                        "latest_step_metrics": latest,
                    }
                    write_json(output / "status.json", progress)
                    print(
                        f"[{sample_number}/{len(selected_indices)}] "
                        f"ASR@{record_steps[-1]}={latest['conditional_attack_success_rate']:.4f} "
                        f"Top5-evict={latest['conditional_top5_eviction_rate']:.4f} "
                        f"sec/sample={elapsed / sample_number:.2f}",
                        flush=True,
                    )
    except BaseException as error:
        failed = {
            **common,
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(output / "status.json", failed)
        raise

    elapsed = time.monotonic() - started
    summary = {
        **common,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(len(selected_indices), 1),
        "metrics_by_step": {str(step): aggregates[step].result() for step in record_steps},
    }
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", summary)
    print(json.dumps(summary["metrics_by_step"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
