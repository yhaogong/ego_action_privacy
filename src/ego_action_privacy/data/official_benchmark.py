"""Build the official 278-class Ego-Exo4D Keystep benchmark manifests.

The official baseline repository publishes the exact v2 ego train/val split as
``path,label_id`` CSV files.  This module joins those rows to the audited raw
manifests by take name and the unmodified start/end timestamps.  Every row and
label must match uniquely; partial or approximate joins are rejected.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "https://github.com/EGO4D/ego-exo4d-keystep"
OFFICIAL_SPLIT_URL = (
    "https://raw.githubusercontent.com/EGO4D/ego-exo4d-keystep/refs/heads/main/"
    "fine_grained/data/files_v2/ego/{split}.csv"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _parse_official_path(value: str, expected_split: str) -> tuple[str, float, float]:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ValueError(f"Invalid official relative path: {value!r}")
    split, take_name, filename = path.parts
    if split != expected_split or not filename.startswith("start") or not filename.endswith(".mp4"):
        raise ValueError(f"Unexpected official split/path: {value!r}")
    timing = filename[5:-4]
    try:
        start_text, remainder = timing.split("_end", 1)
        end_text, camera = remainder.split("_", 1)
        start, end = float(start_text), float(end_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Could not parse official clip timing: {value!r}") from error
    if not camera or start < 0 or end <= start:
        raise ValueError(f"Invalid official clip timing/camera: {value!r}")
    return take_name, start, end


def _read_official_csv(path: Path, split: str) -> list[tuple[str, int, tuple[str, float, float]]]:
    result = []
    seen_paths = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected path,label_id")
            relative_path, label_text = row
            if relative_path in seen_paths:
                raise ValueError(f"{path}:{line_number}: duplicate path {relative_path!r}")
            seen_paths.add(relative_path)
            try:
                label_id = int(label_text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid label_id {label_text!r}") from error
            if label_id < 0:
                raise ValueError(f"{path}:{line_number}: negative label_id")
            result.append((relative_path, label_id, _parse_official_path(relative_path, split)))
    if not result:
        raise ValueError(f"Official split CSV is empty: {path}")
    return result


def _source_index(rows: object, split: str) -> dict[tuple[str, float, float], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"Raw {split} manifest must be a list")
    index: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            raise ValueError(f"Invalid raw {split} manifest row")
        video = row.get("video")
        take_name = row.get("take_name")
        if not isinstance(video, dict) or not isinstance(take_name, str):
            raise ValueError(f"Raw {split} row lacks take/video metadata")
        try:
            key = (take_name, float(video["clip_start"]), float(video["clip_end"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Raw {split} row has invalid timing") from error
        if key in index:
            raise ValueError(f"Ambiguous raw manifest key: {key!r}")
        index[key] = row
    return index


def build_official_benchmark_manifests(
    train_csv: str | Path,
    val_csv: str | Path,
    raw_train_manifest: str | Path,
    raw_val_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_classes: int = 278,
) -> dict[str, Any]:
    """Create strict official-v2 train/val manifests and a label map."""
    csv_paths = {
        "train": Path(train_csv).expanduser().resolve(),
        "val": Path(val_csv).expanduser().resolve(),
    }
    raw_paths = {
        "train": Path(raw_train_manifest).expanduser().resolve(),
        "val": Path(raw_val_manifest).expanduser().resolve(),
    }
    destination = Path(output_dir).expanduser().resolve()
    official = {split: _read_official_csv(path, split) for split, path in csv_paths.items()}
    source = {split: _read_json(path) for split, path in raw_paths.items()}
    indices = {split: _source_index(source[split], split) for split in ("train", "val")}

    manifests: dict[str, list[dict[str, Any]]] = {}
    label_semantics: dict[int, tuple[str, str]] = {}
    counts: dict[str, Counter[int]] = {}
    usable_counts: dict[str, Counter[int]] = {}
    for split in ("train", "val"):
        rows = []
        split_counts: Counter[int] = Counter()
        split_usable: Counter[int] = Counter()
        for official_path, label_id, key in official[split]:
            source_row = indices[split].get(key)
            if source_row is None:
                raise ValueError(f"Official {split} row has no exact raw-manifest match: {official_path}")
            action = source_row.get("action")
            if not isinstance(action, dict) or action.get("label") is None or not action.get("name"):
                raise ValueError(f"Matched row lacks action semantics: {official_path}")
            semantics = (str(action["label"]), str(action["name"]))
            previous = label_semantics.setdefault(label_id, semantics)
            if previous != semantics:
                raise ValueError(
                    f"Official label_id {label_id} maps to conflicting actions: {previous!r}, {semantics!r}"
                )
            row = copy.deepcopy(source_row)
            row["sample_id"] = f"official278_{split}_{len(rows):06d}"
            row["action"] = {
                "label": semantics[0],
                "name": semantics[1],
                "class_index": label_id,
            }
            row["official_benchmark"] = {
                "version": "v2",
                "ego_segment_name": official_path,
                "label_id": label_id,
            }
            rows.append(row)
            split_counts[label_id] += 1
            if row.get("sample_usable") is True:
                split_usable[label_id] += 1
        manifests[split] = rows
        counts[split] = split_counts
        usable_counts[split] = split_usable

    expected_ids = set(range(expected_classes))
    if set(label_semantics) != expected_ids:
        missing = sorted(expected_ids - set(label_semantics))
        extra = sorted(set(label_semantics) - expected_ids)
        raise ValueError(f"Expected label IDs 0..{expected_classes - 1}; missing={missing}, extra={extra}")
    raw_to_label: dict[str, int] = {}
    for label_id, (raw_class, _) in label_semantics.items():
        if raw_class in raw_to_label:
            raise ValueError(f"step_unique_id {raw_class} maps to multiple official labels")
        raw_to_label[raw_class] = label_id

    classes = [
        {
            "class_index": label_id,
            "official_label_id": label_id,
            "raw_class": label_semantics[label_id][0],
            "action_name": label_semantics[label_id][1],
            "train_count": counts["train"][label_id],
            "val_count": counts["val"][label_id],
            "train_usable_count": usable_counts["train"][label_id],
            "val_usable_count": usable_counts["val"][label_id],
        }
        for label_id in range(expected_classes)
    ]
    split_stats = {
        split: {
            "official_rows": len(manifests[split]),
            "matched_rows": len(manifests[split]),
            "usable_rows": sum(row.get("sample_usable") is True for row in manifests[split]),
            "observed_classes": len(counts[split]),
        }
        for split in ("train", "val")
    }
    outputs = {
        "train_manifest": destination / "keystep_train.json",
        "val_manifest": destination / "keystep_val.json",
        "class_map": destination / "class_map.json",
        "label_mapping": destination / "label_mapping.csv",
        "official_train_csv": destination / "official_train.csv",
        "official_val_csv": destination / "official_val.csv",
        "report": destination / "manifest_report.json",
    }
    mapping_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        mapping_buffer,
        fieldnames=["label_id", "step_unique_id", "step_name", "train_count", "val_count"],
    )
    writer.writeheader()
    writer.writerows({
        "label_id": row["class_index"],
        "step_unique_id": row["raw_class"],
        "step_name": row["action_name"],
        "train_count": row["train_count"],
        "val_count": row["val_count"],
    } for row in classes)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "official Ego-Exo4D Fine-grained Keystep Recognition v2, ego-only",
        "num_classes": expected_classes,
        "splits": split_stats,
        "all_official_rows_matched_exactly": True,
        "all_rows_usable": all(
            split_stats[split]["official_rows"] == split_stats[split]["usable_rows"]
            for split in split_stats
        ),
        "join_key": "exact (take_name, start_time, end_time)",
        "pixel_source": (
            "official downscaled/448 full-take ego RGB decoded at the official benchmark start/end times; "
            "not separately encoded keystep_benchmark/clips_448p MP4 files"
        ),
        "label_mapping_provenance": (
            "inferred consistently from official baseline CSV label_id and released raw step_unique_id/name; "
            "the official downloader release does not currently ship label_mapping.csv"
        ),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_split_urls": {split: OFFICIAL_SPLIT_URL.format(split=split) for split in csv_paths},
        "sources": {
            "official_csv": {
                split: {"path": str(path), "sha256": _sha256(path)} for split, path in csv_paths.items()
            },
            "raw_manifest": {
                split: {"path": str(path), "sha256": _sha256(path)} for split, path in raw_paths.items()
            },
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(outputs["train_manifest"], manifests["train"])
    _write_json(outputs["val_manifest"], manifests["val"])
    _write_json(outputs["class_map"], {"num_classes": expected_classes, "classes": classes})
    _write_text(outputs["label_mapping"], mapping_buffer.getvalue())
    _write_text(outputs["official_train_csv"], csv_paths["train"].read_text(encoding="utf-8"))
    _write_text(outputs["official_val_csv"], csv_paths["val"].read_text(encoding="utf-8"))
    _write_json(outputs["report"], report)
    return report
