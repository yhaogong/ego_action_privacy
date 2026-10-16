#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from ego_action_privacy.config import load_config, set_seed, torch_dtype, write_json
from ego_action_privacy.data import EgoKeystepDataset, collate_samples
from ego_action_privacy.targets import InternVideo2Classifier
from ego_action_privacy.targets.text_cache import load_text_cache, read_class_map


class DistributedEvalSampler(Sampler[int]):
    """Disjoint evaluation shards without DistributedSampler padding."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def unwrap(model: nn.Module) -> InternVideo2Classifier:
    return model.module if isinstance(model, DistributedDataParallel) else model


def optimizer_groups(model: InternVideo2Classifier, config: dict) -> list[dict]:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        section = "head" if name.startswith("classifier.") else "backbone"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups.setdefault((section, use_decay), []).append(parameter)
    result = []
    for (section, use_decay), parameters in groups.items():
        result.append({
            "params": parameters,
            "lr": float(config[f"{section}_learning_rate"]),
            "weight_decay": float(config["weight_decay"]) if use_decay else 0.0,
            "group_name": f"{section}_{'decay' if use_decay else 'no_decay'}",
        })
    return result


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int, minimum_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return max(step, 1) / warmup_steps
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_dtype: torch.dtype,
    num_classes: int,
) -> dict[str, float | int]:
    module = unwrap(model)
    module.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=compute_dtype):
            logits = module(video)
        logits = logits.float()
        loss = F.cross_entropy(logits, labels, reduction="sum")
        top = logits.topk(min(5, num_classes), dim=1).indices
        totals += torch.tensor([
            labels.numel(),
            float(loss),
            int(top[:, :1].eq(labels[:, None]).any(dim=1).sum()),
            int(top.eq(labels[:, None]).any(dim=1).sum()),
        ], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    samples, loss_sum, top1, top5 = totals.tolist()
    return {
        "samples": int(samples),
        "loss": loss_sum / samples,
        "top1_accuracy": top1 / samples,
        "top5_accuracy": top5 / samples,
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    next_batch: int,
    global_step: int,
    best_top1: float,
    config: dict,
) -> dict:
    return {
        "schema_version": 1,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_top1": best_top1,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for the official Ego-Exo4D 278-way recognizer")
    parser.add_argument("--config", default="configs/recognizer_finetune.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Optimizer-step limit for a smoke or bounded run")
    parser.add_argument("--limit-train", type=int, help="Explicit training subset for smoke tests")
    parser.add_argument("--limit-val", type=int, help="Explicit validation subset for smoke tests")
    args = parser.parse_args()
    config = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl")
    set_seed(int(config["train"]["seed"]) + rank)
    run_dir = Path(args.run_dir).expanduser().resolve()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    data_config = config["data"]
    train_dataset = EgoKeystepDataset(
        config["paths"]["train_manifest"],
        num_frames=int(data_config["num_frames"]),
        output_size=int(data_config["image_size"]),
        availability="valid",
        limit=args.limit_train,
    )
    val_dataset = EgoKeystepDataset(
        config["paths"]["val_manifest"],
        num_frames=int(data_config["num_frames"]),
        output_size=int(data_config["image_size"]),
        availability="valid",
        limit=args.limit_val,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["train"]["seed"]),
    )
    val_sampler = DistributedEvalSampler(len(val_dataset), rank, world_size)
    loader_options = dict(
        batch_size=int(data_config["batch_size_per_gpu"]),
        num_workers=int(data_config["num_workers_per_gpu"]),
        pin_memory=True,
        collate_fn=collate_samples,
        persistent_workers=int(data_config["num_workers_per_gpu"]) > 0,
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, drop_last=False, **loader_options)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **loader_options)
    num_classes = int(data_config["num_classes"])
    raw_classes, _ = read_class_map(config["paths"]["class_map"])
    if len(raw_classes) != num_classes:
        raise ValueError(f"Class map has {len(raw_classes)} classes, expected {num_classes}")

    model_config = config["model"]
    parameter_dtype = torch_dtype(model_config["parameter_dtype"])
    compute_dtype = torch_dtype(model_config["compute_dtype"])
    model = InternVideo2Classifier(
        config["paths"]["model_dir"],
        num_classes,
        device=device,
        parameter_dtype=parameter_dtype,
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
        head_dropout=float(model_config["head_dropout"]),
    )
    if bool(model_config["initialize_head_from_text"]):
        text_features = load_text_cache(config["paths"]["text_cache"], config["paths"]["class_map"])
        model.initialize_classifier(text_features)
        del text_features

    train_config = config["train"]
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, train_config),
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
    accumulation_steps = int(train_config["accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = updates_per_epoch * int(train_config["epochs"])
    warmup_steps = round(total_steps * float(train_config["warmup_ratio"]))
    minimum_ratio = float(train_config["minimum_learning_rate_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, warmup_steps, minimum_ratio),
    )
    start_epoch = 0
    resume_batch = 0
    global_step = 0
    best_top1 = 0.0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported recognizer checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"])
        resume_batch = int(payload["next_batch"])
        global_step = int(payload["global_step"])
        best_top1 = float(payload["best_top1"])
        del payload
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    started = time.time()
    status = {
        "complete": False,
        "task": "official Ego-Exo4D v2 ego 278-way supervised recognition fine-tuning",
        "world_size": world_size,
        "train_coverage": train_dataset.coverage,
        "val_coverage": val_dataset.coverage,
        "effective_batch_size": int(data_config["batch_size_per_gpu"]) * world_size * accumulation_steps,
        "total_planned_updates": total_steps,
        "config": config,
        "load_report": unwrap(model).load_report,
        "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
    }
    if rank == 0:
        write_json(run_dir / "status.json", status)
    metrics_path = run_dir / "train_metrics.jsonl"
    log_stream = metrics_path.open("a", encoding="utf-8") if rank == 0 else contextlib.nullcontext()
    stop = False
    with log_stream as metrics:
        for epoch in range(start_epoch, int(train_config["epochs"])):
            train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            first_batch = resume_batch if epoch == start_epoch else 0
            accumulated_loss = 0.0
            accumulated_samples = 0
            group_size = 0
            group_start = first_batch
            for batch_index, batch in enumerate(train_loader):
                if batch_index < first_batch:
                    continue
                offset = batch_index - group_start
                if offset % accumulation_steps == 0:
                    group_size = min(accumulation_steps, len(train_loader) - batch_index)
                update_now = offset % accumulation_steps == group_size - 1
                video = batch["video"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                sync_context = contextlib.nullcontext() if update_now or not distributed else model.no_sync()
                with sync_context:
                    with torch.autocast("cuda", dtype=compute_dtype):
                        logits = model(video)
                        loss = F.cross_entropy(
                            logits.float(), labels,
                            label_smoothing=float(train_config["label_smoothing"]),
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    (loss / group_size).backward()
                accumulated_loss += float(loss.detach()) * labels.numel()
                accumulated_samples += labels.numel()
                if not update_now:
                    continue
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["gradient_clip_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Nonfinite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduler.step()
                reduced = torch.tensor(
                    [accumulated_loss, accumulated_samples], device=device, dtype=torch.float64
                )
                if distributed:
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                if rank == 0 and (
                    global_step == 1 or global_step % int(train_config["log_every_updates"]) == 0
                ):
                    record = {
                        "type": "train",
                        "epoch": epoch,
                        "next_batch": batch_index + 1,
                        "step": global_step,
                        "loss": reduced[0].item() / reduced[1].item(),
                        "gradient_norm": float(gradient_norm),
                        "learning_rates": {
                            group["group_name"]: group["lr"] for group in optimizer.param_groups
                        },
                        "elapsed_seconds": time.time() - started,
                    }
                    metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                accumulated_loss = 0.0
                accumulated_samples = 0
                if global_step % int(train_config["checkpoint_every_updates"]) == 0:
                    if rank == 0:
                        atomic_torch_save(
                            checkpoint_payload(
                                model, optimizer, scheduler,
                                epoch=epoch,
                                next_batch=batch_index + 1,
                                global_step=global_step,
                                best_top1=best_top1,
                                config=config,
                            ),
                            run_dir / "checkpoint_latest.pt",
                        )
                    if distributed:
                        dist.barrier()
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
            resume_batch = 0
            validation = evaluate(model, val_loader, device, compute_dtype, num_classes)
            if rank == 0:
                record = {"type": "validation", "epoch": epoch, "step": global_step, **validation}
                metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
                is_best = float(validation["top1_accuracy"]) > best_top1
                best_top1 = max(best_top1, float(validation["top1_accuracy"]))
                payload = checkpoint_payload(
                    model, optimizer, scheduler,
                    epoch=epoch + 1,
                    next_batch=0,
                    global_step=global_step,
                    best_top1=best_top1,
                    config=config,
                )
                atomic_torch_save(payload, run_dir / "checkpoint_latest.pt")
                if is_best:
                    atomic_torch_save(
                        {
                            "schema_version": 1,
                            "model": payload["model"],
                            "epoch": epoch,
                            "global_step": global_step,
                            "validation": validation,
                            "config": config,
                        },
                        run_dir / "checkpoint_best.pt",
                    )
            if distributed:
                best_tensor = torch.tensor(best_top1, device=device)
                dist.broadcast(best_tensor, src=0)
                best_top1 = float(best_tensor)
                dist.barrier()
            if stop:
                break

    status.update({
        "complete": not stop and global_step >= total_steps,
        "bounded_run_complete": stop,
        "global_step": global_step,
        "best_top1_accuracy": best_top1,
        "latest_checkpoint": str(run_dir / "checkpoint_latest.pt"),
        "best_checkpoint": str(run_dir / "checkpoint_best.pt"),
        "elapsed_seconds": time.time() - started,
    })
    if rank == 0:
        write_json(run_dir / "status.json", status)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
