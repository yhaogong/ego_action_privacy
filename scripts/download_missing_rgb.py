#!/usr/bin/env python3
"""Plan or download only the ego RGB files selected by video_audit.json.

Run this script with the environment containing the official `ego4d` package
(on the current server: `/data2/yhao/envs/egoexo/bin/python`). Unlike the broad
`--views ego` CLI filter, selection is by exact audited relative path, so exo,
SLAM, and preview streams are not downloaded accidentally.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from pathlib import Path

from ego4d.internal.download.cli import _create_pathmgr, map_all
from ego4d.internal.download.manifest import manifest_loads, PathSpecification


OFFICIAL_MANIFEST = "s3://ego4d-consortium-sharing/egoexo-public/v2/downscaled_takes/448/manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--video-audit", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--s3-profile", default="egoexo")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--yes", action="store_true", help="Perform downloads; otherwise only plan")
    parser.add_argument(
        "--include-unique-official-aria-fallbacks",
        action="store_true",
        help="For unresolved takes, include only a unique official Aria *_214-1.mp4 RGB path",
    )
    parser.add_argument("--overrides", help="Write explicit builder overrides for unique official Aria fallbacks")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    audit = json.loads(Path(args.video_audit).read_text(encoding="utf-8"))
    if not isinstance(audit, list):
        raise ValueError("video_audit.json must be a list")
    unresolved = []
    desired: dict[str, str] = {}
    for row in audit:
        if row.get("exists"):
            continue
        candidates = row.get("candidate_paths")
        if not isinstance(candidates, list) or not candidates:
            unresolved.append({"split": row.get("split"), "take_uid": row.get("take_uid")})
            continue
        candidate = Path(candidates[0]).resolve()
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Candidate lies outside data root: {candidate}") from error
        previous = desired.setdefault(relative, row["take_uid"])
        if previous != row["take_uid"]:
            raise ValueError(f"Two take UIDs map to the same path: {relative}")

    path_manager = _create_pathmgr(args.s3_profile)
    entries = manifest_loads(path_manager.open(OFFICIAL_MANIFEST).read())
    entries_by_uid = {entry.uid: entry for entry in entries}
    fallback_overrides = {}
    still_unresolved = []
    for row in unresolved:
        entry = entries_by_uid.get(row["take_uid"])
        candidates = [] if entry is None else [
            path for path in entry.paths
            if re.fullmatch(r"aria.*_214-1\.mp4", Path(path.relative_path).name, flags=re.IGNORECASE)
        ]
        if args.include_unique_official_aria_fallbacks and len(candidates) == 1:
            path = candidates[0]
            desired[path.relative_path] = row["take_uid"]
            fallback_overrides[row["take_uid"]] = {
                "relative_path": path.relative_path,
                "camera": Path(path.relative_path).name.rsplit("_214-1.mp4", 1)[0],
                "stream": "rgb",
                "selection_reason": "unique official Aria 214-1 RGB path for this annotated take_uid",
                "source_manifest": OFFICIAL_MANIFEST,
            }
        else:
            still_unresolved.append({
                **row,
                "official_manifest_entry_present": entry is not None,
                "official_manifest_path_count": len(entry.paths) if entry is not None else 0,
                "official_aria_rgb_candidate_count": len(candidates),
            })
    if args.overrides:
        override_path = Path(args.overrides).expanduser().resolve()
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(json.dumps({
            "schema_version": 1,
            "source_manifest": OFFICIAL_MANIFEST,
            "selection_policy": "unique official Aria *_214-1.mp4 path for annotated take_uid",
            "takes": fallback_overrides,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    official = {path.relative_path: path for entry in entries for path in entry.paths}
    missing_from_official = sorted(set(desired) - set(official))
    selected = [official[path] for path in sorted(desired) if path in official]
    already_present = [path for path in selected if (root / path.relative_path).is_file() and (root / path.relative_path).stat().st_size == path.size]
    downloads = [path for path in selected if path not in already_present]
    bytes_to_download = sum(path.size or 0 for path in downloads)
    report = {
        "official_manifest": OFFICIAL_MANIFEST,
        "data_root": str(root),
        "audited_missing_or_unresolved_takes": len({row["take_uid"] for row in audit if not row.get("exists")}),
        "resolved_target_paths": len(desired),
        "initial_unresolved_take_count": len(unresolved),
        "official_unique_aria_fallback_count": len(fallback_overrides),
        "unresolved_take_count": len(still_unresolved),
        "unresolved_takes": still_unresolved,
        "matched_official_paths": len(selected),
        "missing_from_official_manifest": missing_from_official,
        "already_present_with_official_size": len(already_present),
        "files_to_download": len(downloads),
        "bytes_to_download": bytes_to_download,
        "gib_to_download": bytes_to_download / 1024**3,
        "selection_policy": (
            "exact first candidate path from video_audit.json, plus explicitly enabled unique official Aria 214-1 fallbacks; "
            "no preview, SLAM, or exo streams"
            if args.include_unique_official_aria_fallbacks
            else "exact first candidate path from video_audit.json; no preview, SLAM, or exo streams"
        ),
        "download_requested": args.yes,
    }
    destination = Path(args.report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "unresolved_takes"}, ensure_ascii=False, indent=2), flush=True)
    if missing_from_official:
        print("Some resolved audit paths are absent from the official manifest; refusing partial exact download.")
        return 2
    if not args.yes or not downloads:
        print("PLAN ONLY: pass --yes to download." if downloads else "No matched files need downloading.")
        return 0

    for path in downloads:
        (root / path.relative_path).parent.mkdir(parents=True, exist_ok=True)

    def download(path: PathSpecification, downloader):
        output = root / path.relative_path
        try:
            downloader.copy(path.source_path, str(output))
            actual = output.stat().st_size
            if path.size is not None and actual != path.size:
                return None, path, f"size mismatch: expected {path.size}, got {actual}"
            return actual, path, None
        except Exception:
            return None, path, traceback.format_exc()

    successes, failures = map_all(
        downloads,
        map_fn=download,
        num_workers=args.workers,
        s3_profile=args.s3_profile,
        needs_downloader=True,
        progress_on_bytes=True,
        total_bytes=bytes_to_download,
    )
    report.update(
        download_complete=len(failures) == 0 and len(successes) == len(downloads),
        downloaded_files=len(successes),
        download_failures=[{"relative_path": path.relative_path, "error": error} for path, error in failures],
    )
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(successes)}/{len(downloads)} exact RGB files; failures={len(failures)}")
    return 0 if report["download_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
