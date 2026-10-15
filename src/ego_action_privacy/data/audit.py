"""Decode audits for generated manifests."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import read_video_clip


def audit_manifest_decode(
    manifest: str | Path,
    *,
    num_frames: int = 8,
    max_samples: int | None = None,
    one_clip_per_video: bool = False,
) -> dict[str, Any]:
    path = Path(manifest).expanduser().resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list")
    usable = [row for row in rows if row.get("sample_usable") is True]
    if max_samples is not None:
        usable = usable[:max_samples]
    if one_clip_per_video:
        unique = {}
        for row in usable:
            unique.setdefault(row["video"]["path"], row)
        usable = list(unique.values())
    failures = []
    successes = 0
    by_video: dict[str, dict[str, Any]] = defaultdict(lambda: {"attempted": 0, "passed": 0, "failed": 0})
    for row in usable:
        video = row["video"]
        video_path = video["path"]
        by_video[video_path]["attempted"] += 1
        try:
            _, metadata = read_video_clip(
                video_path,
                float(video["clip_start"]),
                float(video["clip_end"]),
                num_frames=num_frames,
                output_size=None,
            )
            successes += 1
            by_video[video_path]["passed"] += 1
            by_video[video_path].update(fps=metadata["fps"], frame_count=metadata["frame_count"])
        except Exception as error:  # report codec and corrupt-file failures without aborting the audit
            by_video[video_path]["failed"] += 1
            failures.append({
                "sample_id": row.get("sample_id"),
                "take_uid": row.get("take_uid"),
                "video_path": video_path,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(path),
        "manifest_rows": len(rows),
        "manifest_usable_rows": sum(row.get("sample_usable") is True for row in rows),
        "attempted_rows": len(usable),
        "decoded_rows": successes,
        "failed_rows": len(failures),
        "decode_complete": len(failures) == 0 and len(usable) == sum(row.get("sample_usable") is True for row in rows),
        "one_clip_per_video": one_clip_per_video,
        "failures": failures,
        "videos": dict(by_video),
    }


def write_audit(report: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
