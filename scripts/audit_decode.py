#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ego_action_privacy.data.audit import audit_manifest_decode, write_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode sampled frames from a generated manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--one-clip-per-video", action="store_true")
    args = parser.parse_args()
    report = audit_manifest_decode(
        args.manifest,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
        one_clip_per_video=args.one_clip_per_video,
    )
    write_audit(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ("failures", "videos")}, indent=2))
    print(f"Failures: {len(report['failures'])}; report: {args.output}")
    return 0 if report["failed_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

