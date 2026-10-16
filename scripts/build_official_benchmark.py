#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ego_action_privacy.data.official_benchmark import build_official_benchmark_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict official 278-class Ego-Exo4D benchmark manifests")
    parser.add_argument("--train-csv", required=True, help="Official baseline repository ego/train.csv")
    parser.add_argument("--val-csv", required=True, help="Official baseline repository ego/val.csv")
    parser.add_argument("--raw-train-manifest", required=True)
    parser.add_argument("--raw-val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_official_benchmark_manifests(
        args.train_csv,
        args.val_csv,
        args.raw_train_manifest,
        args.raw_val_manifest,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
