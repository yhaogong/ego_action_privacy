"""Small configuration helpers without a framework dependency."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(values)}") from error


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
