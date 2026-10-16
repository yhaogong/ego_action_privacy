#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REQUIRED_MODEL_FILES = [
    "InternVideo2-stage2_1b-224p-f4.pt",
    "internvl_c_13b_224px.pth",
    "1B_clip.pth",
    "chinese_alpaca_lora_7b/config.json",
    "chinese_alpaca_lora_7b/tokenizer.model",
    "chinese_alpaca_lora_7b/tokenizer_config.json",
    "chinese_alpaca_lora_7b/special_tokens_map.json",
]


def file_record(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.is_file() else None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local data, model, source, Python and GPU resources")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--internvideo-source", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = Path(args.data_root).expanduser().resolve()
    model = Path(args.model_dir).expanduser().resolve()
    sources = Path(args.internvideo_source).expanduser().resolve()
    # Do not resolve this symlink: venv/bin/python commonly points at the base
    # interpreter, and invoking the resolved target would bypass the venv.
    python = Path(args.python).expanduser().absolute()
    records = [
        file_record(data / "annotations/keystep_train.json"),
        file_record(data / "annotations/keystep_val.json"),
        file_record(data / "takes.json"),
        file_record(data / "takes"),
        *[file_record(model / relative) for relative in REQUIRED_MODEL_FILES],
    ]
    probe = subprocess.run(
        [str(python), "-c", (
            "import json,sys,torch,torchvision; "
            "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
            "'torchvision':torchvision.__version__,'cuda_runtime':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        )],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(sources), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    report = {
        "ready": all(record["exists"] for record in records) and probe.returncode == 0 and commit.returncode == 0,
        "resources": records,
        "python_probe": json.loads(probe.stdout) if probe.returncode == 0 else {"error": probe.stderr.strip()},
        "internvideo_source": str(sources),
        "internvideo_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "expected_internvideo_commit": "3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3",
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] and report["internvideo_commit"] == report["expected_internvideo_commit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
