#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ego_action_privacy.data.manifest import build_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full, unfiltered Ego-Exo4D Keystep manifests")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--takes-json")
    parser.add_argument("--train-ann")
    parser.add_argument("--val-ann")
    parser.add_argument("--video-overrides", help="Audited official-path overrides for takes missing usable metadata")
    parser.add_argument("--resolution", type=int, default=448)
    args = parser.parse_args()
    report = build_manifests(
        args.data_root,
        args.output_dir,
        takes_json=args.takes_json,
        train_annotation=args.train_ann,
        val_annotation=args.val_ann,
        video_overrides=args.video_overrides,
        resolution=args.resolution,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["ready_for_full_data_loading"]:
        print("FULL FILE-PRESENCE AUDIT PASSED (decoding still requires audit_decode.py)")
        return 0
    if report["ready_for_all_valid_samples"]:
        print("ALL VALID ANNOTATIONS HAVE VIDEO; invalid source rows remain retained for audit")
        return 0
    print("AUDIT BLOCKED: all rows retained; inspect issues.json and video_audit.json")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
