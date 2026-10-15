"""Build auditable manifests from raw Ego-Exo4D Keystep annotations.

All annotation rows are retained. Missing videos and invalid annotations are
reported explicitly and never silently turned into a "full" usable dataset.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid step_unique_id: {value!r}")
    result = str(value).strip()
    if not result:
        raise ValueError("Empty step_unique_id")
    return result


def _id_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value, value)


def _vocabulary(root: object) -> dict[str, str]:
    if not isinstance(root, dict) or not isinstance(root.get("vocabulary"), dict):
        raise ValueError("Expected raw Keystep vocabulary mapping")
    result: dict[str, str] = {}
    for raw, name in root["vocabulary"].items():
        key = _raw_id(raw)
        if not isinstance(name, str) or not name.strip() or key in result:
            raise ValueError(f"Invalid/duplicate vocabulary entry: {key!r}")
        result[key] = name.strip()
    if not result:
        raise ValueError("Keystep vocabulary is empty")
    return result


def _annotation_takes(root: object) -> dict[str, dict[str, Any]]:
    if not isinstance(root, dict):
        raise ValueError("Expected annotation object")
    value = root.get("annotations")
    if isinstance(value, dict):
        entries = value.items()
    elif isinstance(value, list):
        entries = ((None, item) for item in value)
    else:
        raise ValueError("Expected annotations grouped by take_uid")
    result: dict[str, dict[str, Any]] = {}
    for outer_uid, take in entries:
        if not isinstance(take, dict) or not isinstance(take.get("segments"), list):
            raise ValueError("Each annotation take must contain a segments list")
        inner_uid = take.get("take_uid")
        if inner_uid and outer_uid and str(inner_uid) != str(outer_uid):
            raise ValueError(f"Conflicting annotation take_uid: {outer_uid}, {inner_uid}")
        uid = str(inner_uid or outer_uid or "").strip()
        if not uid or uid in result:
            raise ValueError(f"Missing/duplicate annotation take_uid: {uid!r}")
        result[uid] = take
    return result


def _metadata_takes(value: object) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("takes"), list):
        value = value["takes"]
    if isinstance(value, dict):
        entries = value.items()
    elif isinstance(value, list):
        entries = ((None, item) for item in value)
    else:
        raise ValueError("takes.json must be a take list or take_uid mapping")
    result: dict[str, dict[str, Any]] = {}
    for outer_uid, take in entries:
        if not isinstance(take, dict):
            raise ValueError("Invalid takes.json entry")
        inner_uid = take.get("take_uid")
        if inner_uid and outer_uid and str(inner_uid) != str(outer_uid):
            raise ValueError("Conflicting metadata take_uid")
        uid = str(inner_uid or outer_uid or "").strip()
        if not uid or uid in result:
            raise ValueError(f"Missing/duplicate metadata take_uid: {uid!r}")
        result[uid] = take
    return result


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing metadata relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe metadata relative path: {value!r}")
    return path


def _resolve_video(take: dict[str, Any] | None, data_root: Path, resolution: int) -> tuple[dict[str, Any], str | None]:
    """Resolve one ego RGB stream without ever guessing an exocentric view."""
    if take is None:
        raise ValueError("take_uid absent from takes.json")
    videos = take.get("frame_aligned_videos")
    if not isinstance(videos, dict):
        raise ValueError("Missing frame_aligned_videos metadata")
    cameras = (take.get("capture") or {}).get("cameras")
    warning = None
    metadata_aria_ego = False
    if isinstance(cameras, list) and cameras:
        ego_ids = {
            str(camera["cam_id"])
            for camera in cameras
            if isinstance(camera, dict) and camera.get("is_ego") is True and camera.get("cam_id")
        }
        metadata_aria_ego = any(
            isinstance(camera, dict)
            and camera.get("is_ego") is True
            and str(camera.get("device_type", "")).lower() == "aria"
            for camera in cameras
        )
    else:
        ego_ids = {str(key) for key in videos if str(key).lower().startswith("aria")}
        warning = "camera metadata unavailable; unique Aria RGB naming fallback used"
    camera_ids = set(ego_ids)
    if metadata_aria_ego and not camera_ids.intersection(videos):
        # A few official records call the capture camera `aria01` but use the
        # frame-aligned key `Aria`. The 214-1 stream is still explicit RGB.
        aria_keys = {str(key) for key in videos if str(key).lower().startswith("aria")}
        if len(aria_keys) == 1:
            camera_ids.update(aria_keys)
            warning = "ego camera/frame-aligned ID mismatch; unique Aria RGB stream used"
    candidates = []
    for camera in sorted(camera_ids):
        streams = videos.get(camera)
        if not isinstance(streams, dict):
            continue
        for stream_key, stream in streams.items():
            if not isinstance(stream, dict) or not stream.get("relative_path"):
                continue
            names = {str(stream_key), str(stream.get("readable_stream_id", "")), str(stream.get("stream_id", ""))}
            is_aria = camera.lower().startswith("aria")
            if (is_aria and names.intersection({"rgb", "214-1"})) or (
                not is_aria and names.intersection({"rgb", "0"})
            ):
                candidates.append((camera, str(stream_key), stream, is_aria))
    aria = [candidate for candidate in candidates if candidate[3]]
    preferred = aria or candidates
    if len(preferred) != 1:
        names = [(item[0], item[1]) for item in candidates]
        raise ValueError(f"Expected one preferred ego RGB stream, found {len(preferred)}; candidates={names}")
    camera, stream_key, stream, _ = preferred[0]
    take_root = data_root / _relative_path(take.get("root_dir"))
    original = take_root / _relative_path(stream["relative_path"])
    downscaled = original.parent / "downscaled" / str(resolution) / original.name
    options = [(downscaled, f"downscaled_{resolution}"), (original, "original")]
    selected, variant = options[0]
    exists = False
    for path, candidate_variant in options:
        if path.is_file() and path.stat().st_size > 0:
            selected, variant, exists = path, candidate_variant, True
            break
    return {
        "camera": camera,
        "stream": stream_key,
        "path": str(selected),
        "exists": exists,
        "source_variant": variant if exists else "missing",
        "candidate_paths": [str(path) for path, _ in options],
        "alternative_ego_streams": [[item[0], item[1]] for item in candidates if item != preferred[0]],
    }, warning


def _resolve_override(uid: str, override: object, data_root: Path) -> dict[str, Any]:
    if not isinstance(override, dict):
        raise ValueError(f"Invalid video override for {uid}")
    relative = _relative_path(override.get("relative_path"))
    reason = override.get("selection_reason")
    source = override.get("source_manifest")
    if not isinstance(reason, str) or not reason or not isinstance(source, str) or not source:
        raise ValueError(f"Video override for {uid} lacks explicit selection provenance")
    path = data_root / relative
    return {
        "camera": override.get("camera", "official_unique_aria"),
        "stream": "rgb",
        "path": str(path),
        "exists": path.is_file() and path.stat().st_size > 0,
        "source_variant": "audited_official_override",
        "candidate_paths": [str(path)],
        "alternative_ego_streams": [],
        "override_selection_reason": reason,
        "override_source_manifest": source,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _safe_json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifests(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    takes_json: str | Path | None = None,
    train_annotation: str | Path | None = None,
    val_annotation: str | Path | None = None,
    video_overrides: str | Path | None = None,
    resolution: int = 448,
) -> dict[str, Any]:
    """Build full manifests and return the audit report.

    A non-ready report is still a successful manifest generation: bad rows stay
    represented so that coverage problems cannot disappear through filtering.
    """
    root = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    sources = {
        "takes": Path(takes_json or root / "takes.json").expanduser().resolve(),
        "train": Path(train_annotation or root / "annotations/keystep_train.json").expanduser().resolve(),
        "val": Path(val_annotation or root / "annotations/keystep_val.json").expanduser().resolve(),
    }
    roots = {split: _load_json(path) for split, path in sources.items() if split != "takes"}
    vocabularies = {split: _vocabulary(value) for split, value in roots.items()}
    if vocabularies["train"] != vocabularies["val"]:
        raise ValueError("Train/val vocabularies differ; refusing independent label maps")
    vocabulary = vocabularies["train"]
    labels = sorted(vocabulary, key=_id_key)
    indices = {label: index for index, label in enumerate(labels)}
    annotations = {split: _annotation_takes(value) for split, value in roots.items()}
    metadata = _metadata_takes(_load_json(sources["takes"]))
    override_path = Path(video_overrides).expanduser().resolve() if video_overrides else None
    if override_path is not None:
        override_root = _load_json(override_path)
        if not isinstance(override_root, dict) or override_root.get("schema_version") != 1 or not isinstance(override_root.get("takes"), dict):
            raise ValueError("Invalid video override file")
        overrides = override_root["takes"]
    else:
        overrides = {}
    destination.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, Any]] = []
    video_audit: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, int]] = {}
    counts: dict[str, Counter[str]] = {}
    usable_counts: dict[str, Counter[str]] = {}
    overlap = sorted(set(annotations["train"]) & set(annotations["val"]))
    if overlap:
        issues.append({"severity": "error", "reason": "train_val_take_overlap", "take_uids": overlap})

    for split, split_takes in annotations.items():
        rows: list[dict[str, Any]] = []
        split_counts: Counter[str] = Counter()
        split_usable_counts: Counter[str] = Counter()
        valid_count = available_count = 0
        for uid, annotation in split_takes.items():
            take = metadata.get(uid)
            context = {"split": split, "take_uid": uid}
            try:
                video, warning = _resolve_video(take, root, resolution)
                if warning:
                    issues.append({**context, "severity": "warning", "reason": warning})
                if not video["exists"]:
                    issues.append({**context, "severity": "error", "reason": "missing_or_empty_video"})
            except ValueError as error:
                if uid in overrides:
                    video = _resolve_override(uid, overrides[uid], root)
                    issues.append({
                        **context,
                        "severity": "warning",
                        "reason": f"metadata resolution failed ({error}); audited official override used",
                    })
                    if not video["exists"]:
                        issues.append({**context, "severity": "error", "reason": "override_video_missing_or_empty"})
                else:
                    video = {
                        "camera": None,
                        "stream": None,
                        "path": None,
                        "exists": False,
                        "source_variant": "unresolved",
                        "candidate_paths": [],
                        "alternative_ego_streams": [],
                    }
                    issues.append({**context, "severity": "error", "reason": str(error)})
            video_audit.append({**context, **video, "num_segments": len(annotation["segments"])})
            for source_index, source_segment in enumerate(annotation["segments"]):
                sample_id = f"keystep_{split}_{uid}_{source_index:06d}"
                segment = source_segment
                errors: list[str] = []
                if not isinstance(segment, dict):
                    errors.append("segment is not an object")
                    segment = {"raw_segment": segment}
                label: str | None = None
                start: float | None = None
                end: float | None = None
                try:
                    label = _raw_id(segment.get("step_unique_id"))
                    if label not in vocabulary:
                        raise ValueError(f"step_unique_id {label} absent from vocabulary")
                    split_counts[label] += 1
                except ValueError as error:
                    errors.append(str(error))
                try:
                    if any(isinstance(segment.get(key), bool) for key in ("start_time", "end_time")):
                        raise ValueError("boolean clip time")
                    start, end = float(segment["start_time"]), float(segment["end_time"])
                    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                        raise ValueError("expected finite 0 <= start_time < end_time")
                    duration = (take or {}).get("duration_sec")
                    if duration is not None and end > float(duration) + 0.1:
                        raise ValueError(f"end_time {end} exceeds take duration_sec {duration}")
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"invalid interval: {error}")
                    start = start if start is not None and math.isfinite(start) else None
                    end = end if end is not None and math.isfinite(end) else None
                if errors:
                    issues.append({**context, "sample_id": sample_id, "severity": "error", "reasons": errors})
                else:
                    valid_count += 1
                    available_count += int(video["exists"])
                    if video["exists"] and label is not None:
                        split_usable_counts[label] += 1
                name = vocabulary.get(label) if label is not None else None
                if name is not None and segment.get("step_name") not in (None, name):
                    issues.append({
                        **context,
                        "sample_id": sample_id,
                        "severity": "warning",
                        "reason": "step_name differs from vocabulary; canonical vocabulary text used",
                    })
                rows.append({
                    "sample_id": sample_id,
                    "split": split,
                    "take_uid": uid,
                    "take_name": (take or {}).get("take_name") or annotation.get("take_name"),
                    "source_index": source_index,
                    "video": {
                        **{key: value for key, value in video.items() if key not in ("candidate_paths", "alternative_ego_streams")},
                        "clip_start": start,
                        "clip_end": end,
                    },
                    "action": {"label": label, "name": name, "class_index": indices.get(label)},
                    "source_annotation": _safe_json_value(source_segment),
                    "annotation_valid": not errors,
                    "sample_usable": not errors and bool(video["exists"]),
                })
        manifests[split] = rows
        counts[split] = split_counts
        usable_counts[split] = split_usable_counts
        stats[split] = {
            "takes": len(split_takes),
            "source_segments": len(rows),
            "manifest_rows": len(rows),
            "valid_annotation_rows": valid_count,
            "valid_rows_with_video_file": available_count,
            "valid_rows_missing_video_file": valid_count - available_count,
            "observed_classes": len(split_counts),
            "usable_observed_classes": len(split_usable_counts),
        }

    classes = [
        {
            "class_index": indices[label],
            "raw_class": label,
            "action_name": vocabulary[label],
            "train_count": counts["train"][label],
            "val_count": counts["val"][label],
            "train_usable_count": usable_counts["train"][label],
            "val_usable_count": usable_counts["val"][label],
        }
        for label in labels
    ]
    train_labels, val_labels = set(counts["train"]), set(counts["val"])
    missing = sorted({row["take_uid"] for row in video_audit if not row["exists"]})
    error_count = sum(issue["severity"] == "error" for issue in issues)
    outputs = {
        "train_manifest": str(destination / "keystep_train.json"),
        "val_manifest": str(destination / "keystep_val.json"),
        "class_map": str(destination / "class_map.json"),
        "report": str(destination / "manifest_report.json"),
        "video_audit": str(destination / "video_audit.json"),
        "issues": str(destination / "issues.json"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all supplied raw Keystep train/val segments, one ego RGB view per segment",
        "manifest_generation_complete": True,
        "ready_for_full_data_loading": error_count == 0,
        "ready_for_all_valid_samples": all(
            split["valid_rows_missing_video_file"] == 0 for split in stats.values()
        ),
        "video_decode_checked": False,
        "video_check": "selected path is a nonempty file; use audit_decode.py for decoding",
        "vocabulary_size": len(labels),
        "splits": stats,
        "num_segments_total": sum(len(rows) for rows in manifests.values()),
        "train_val_shared_class_count": len(train_labels & val_labels),
        "val_classes_absent_from_train": sorted(val_labels - train_labels, key=_id_key),
        "vocabulary_classes_absent_from_train": sorted(set(labels) - train_labels, key=_id_key),
        "vocabulary_classes_absent_from_both": sorted(set(labels) - train_labels - val_labels, key=_id_key),
        "train_val_overlapping_take_uids": overlap,
        "missing_or_unresolved_video_take_count": len(missing),
        "error_count": error_count,
        "warning_count": len(issues) - error_count,
        "class_index_policy": "one shared map from full vocabulary, numeric step_unique_id order",
        "camera_policy": "unique Aria ego RGB preferred, else unique metadata-marked ego RGB GoPro",
        "video_resolution_policy": f"prefer downscaled/{resolution}; fallback to original; report missing",
        "clip_policy": "unaltered annotated start_time/end_time; no fixed-duration crop",
        "filtering": "none; unusable rows retained and marked sample_usable=false",
        "sources": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in sources.items()},
        "video_overrides": (
            {"path": str(override_path), "sha256": _sha256(override_path), "take_count": len(overrides)}
            if override_path is not None
            else None
        ),
        "outputs": outputs,
    }
    _write_json(Path(outputs["train_manifest"]), manifests["train"])
    _write_json(Path(outputs["val_manifest"]), manifests["val"])
    _write_json(Path(outputs["class_map"]), {"num_classes": len(classes), "classes": classes})
    _write_json(Path(outputs["video_audit"]), video_audit)
    _write_json(Path(outputs["issues"]), issues)
    _write_json(destination / "missing_video_take_uids.json", missing)
    _write_text(destination / "missing_video_take_uids.txt", "".join(uid + "\n" for uid in missing))
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(classes[0]))
    writer.writeheader()
    writer.writerows(classes)
    _write_text(destination / "per_class_counts.csv", table.getvalue())
    _write_json(Path(outputs["report"]), report)
    return report
