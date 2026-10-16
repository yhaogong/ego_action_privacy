#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ego_action_privacy.config import load_config, set_seed, torch_dtype, write_json
from ego_action_privacy.data import EgoKeystepDataset, collate_samples
from ego_action_privacy.losses import LossWeights, protection_loss
from ego_action_privacy.system import ActionPrivacySystem
from ego_action_privacy.targets import InternVideo2CLIP
from ego_action_privacy.targets.text_cache import load_text_cache, read_class_map, save_text_cache
from ego_action_privacy.training import (
    generator_from_config,
    load_generator_checkpoint,
    save_checkpoint,
    trainable_parameter_checksum,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only the bounded U-Net perturbation generator")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial-data", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit smoke-test sample limit")
    parser.add_argument("--max-steps", type=int, help="Override configured max_steps")
    parser.add_argument("--resume", help="Resume generator and optimizer from a project checkpoint")
    parser.add_argument("--temporal-mode", choices=("3d", "framewise"), help="Generator temporal ablation override")
    parser.add_argument("--temporal-weight", type=float, help="Temporal-loss ablation override; use 0 to disable")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.temporal_mode is not None:
        config["generator"]["temporal_mode"] = args.temporal_mode
    if args.temporal_weight is not None:
        if args.temporal_weight < 0:
            parser.error("--temporal-weight must be nonnegative")
        config["loss"]["temporal"] = args.temporal_weight
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    allow_partial = args.allow_partial_data or bool(config["data"].get("allow_partial_data", False))
    availability = "available" if allow_partial else "valid"
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["train"]["seed"]))
    dataset = EgoKeystepDataset(
        config["paths"]["train_manifest"],
        num_frames=int(config["data"]["num_frames"]),
        output_size=int(config["data"]["image_size"]),
        availability=availability,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_samples,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    cache_path = Path(config["paths"]["text_cache"])
    cache_exists = cache_path.is_file()
    target = InternVideo2CLIP(
        config["paths"]["model_dir"],
        device=args.device,
        dtype=torch_dtype(config["target"]["dtype"]),
        gradient_checkpointing=bool(config["target"]["gradient_checkpointing"]),
        load_text_encoder=not cache_exists,
    )
    if cache_exists:
        text_features = load_text_cache(cache_path, config["paths"]["class_map"])
    else:
        _, names = read_class_map(config["paths"]["class_map"])
        text_features = target.encode_text(names, batch_size=int(config["target"]["text_batch_size"]))
        save_text_cache(
            cache_path,
            text_features,
            config["paths"]["class_map"],
            metadata={"model": config["target"]["name"], "text_prefix": "summarize:", "max_tokens": 80},
        )
        target.release_text_encoder()
    resume_payload = None
    if args.resume:
        generator, resume_payload = load_generator_checkpoint(args.resume, args.device)
        if resume_payload["generator_config"] != config["generator"]:
            raise ValueError("Resume checkpoint generator configuration differs from the requested config")
    else:
        generator = generator_from_config(config["generator"]).to(args.device)
    system = ActionPrivacySystem(generator, target, text_features.to(args.device)).to(args.device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    weights = LossWeights(**{key: float(value) for key, value in config["loss"].items()})
    max_steps = args.max_steps if args.max_steps is not None else config["train"].get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    starting_checksum = trainable_parameter_checksum(generator)
    status = {
        "complete": False,
        "partial_data": allow_partial,
        "dataset_coverage": dataset.coverage,
        "config": config,
        "device": args.device,
        "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
        "target_load_report": target.load_report,
        "text_cache": str(cache_path.resolve()),
    }
    write_json(run_dir / "status.json", status)
    log_path = run_dir / "train_metrics.jsonl"
    global_step = int(resume_payload["step"]) if resume_payload is not None else 0
    start_epoch = int(resume_payload["epoch"]) + 1 if resume_payload is not None else 0
    last_epoch = start_epoch - 1
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, int(config["train"]["epochs"])):
            last_epoch = epoch
            for batch in loader:
                clean = batch["video"].to(args.device, non_blocking=True)
                labels = batch["labels"].to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = system(clean)
                terms = protection_loss(
                    output["logits"],
                    labels,
                    clean,
                    output["protected"],
                    output["delta"],
                    weights,
                )
                if not torch.isfinite(terms["total"]):
                    raise RuntimeError("Nonfinite training loss")
                terms["total"].backward()
                if any(parameter.grad is not None for parameter in target.parameters()):
                    raise RuntimeError("Frozen InternVideo2 target accumulated parameter gradients")
                gradient_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Nonfinite generator gradient")
                optimizer.step()
                global_step += 1
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    **{name: float(value.detach()) for name, value in terms.items()},
                    "generator_gradient_norm": float(gradient_norm),
                    "delta_linf": float(output["delta"].detach().abs().amax()),
                    "batch_sample_ids": [item["sample_id"] for item in batch["metadata"]],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if global_step == 1 or global_step % int(config["train"]["log_every"]) == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if global_step % int(config["train"]["checkpoint_every"]) == 0:
                    save_checkpoint(
                        run_dir / f"checkpoint_step_{global_step:07d}.pt",
                        generator,
                        optimizer,
                        step=global_step,
                        epoch=epoch,
                        generator_config=config["generator"],
                        full_config=config,
                    )
                if max_steps is not None and global_step >= max_steps:
                    break
            if max_steps is not None and global_step >= max_steps:
                break
    final_checkpoint = run_dir / "checkpoint_final.pt"
    save_checkpoint(
        final_checkpoint,
        generator,
        optimizer,
        step=global_step,
        epoch=last_epoch,
        generator_config=config["generator"],
        full_config=config,
    )
    ending_checksum = trainable_parameter_checksum(generator)
    if global_step and starting_checksum == ending_checksum:
        raise RuntimeError("Generator parameters did not change")
    status.update({
        "complete": True,
        "steps": global_step,
        "generator_parameters_updated": starting_checksum != ending_checksum,
        "starting_parameter_checksum": starting_checksum,
        "ending_parameter_checksum": ending_checksum,
        "final_checkpoint": str(final_checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
    })
    write_json(run_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
