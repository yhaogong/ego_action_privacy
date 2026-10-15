"""Video dataset for generated Keystep manifests."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def sample_frame_indices(
    start_sec: float,
    end_sec: float,
    fps: float,
    frame_count: int,
    num_frames: int = 8,
) -> list[int]:
    """Uniformly sample an annotated half-open time interval."""
    values = (start_sec, end_sec, fps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite clip timing/FPS")
    if fps <= 0 or frame_count <= 0 or start_sec < 0 or end_sec <= start_sec or num_frames <= 0:
        raise ValueError("Invalid interval, video metadata, or frame count")
    first = math.ceil(start_sec * fps)
    last = min(frame_count - 1, math.ceil(end_sec * fps) - 1)
    if first > last or first >= frame_count:
        raise ValueError("Clip contains no valid video frames")
    return torch.linspace(first, last, num_frames).round().long().tolist()


def read_video_clip(
    path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    num_frames: int = 8,
    output_size: int | None = 224,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode RGB `[T,3,H,W]` in `[0,1]` and return decode metadata."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Video decoding requires opencv-python-headless") from error
    video_path = Path(path)
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(start_sec, end_sec, fps, frame_count, num_frames)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode frame {index}: {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
    finally:
        capture.release()
    video = torch.stack(frames).float().div_(255.0)
    if output_size is not None and video.shape[-2:] != (output_size, output_size):
        video = F.interpolate(
            video,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return video, {"indices": indices, "fps": fps, "frame_count": frame_count}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return value


def _validate_row(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Manifest row is not an object")
    video, action = row.get("video"), row.get("action")
    if not isinstance(video, dict) or not isinstance(action, dict):
        raise ValueError("Manifest row requires video/action objects")
    if not isinstance(action.get("class_index"), int) or isinstance(action.get("class_index"), bool):
        raise ValueError(f"Invalid class_index in {row.get('sample_id')}")
    if not isinstance(action.get("name"), str) or not action["name"].strip():
        raise ValueError(f"Invalid action name in {row.get('sample_id')}")
    return row


class EgoKeystepDataset(Dataset[dict[str, Any]]):
    """Keystep clips with an explicit partial-data policy.

    `availability="error"` refuses any non-usable row. `"valid"` excludes
    explicitly invalid annotations but refuses missing video for every valid row.
    `availability="available"` is the deliberate small-scale/partial-data mode;
    the dropped count is exposed through `coverage` and should be reported.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        num_frames: int = 8,
        output_size: int | None = 224,
        availability: Literal["error", "valid", "available", "all"] = "error",
        decode_video: bool = True,
        limit: int | None = None,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        source_rows = [_validate_row(row) for row in _read_rows(self.manifest)]
        usable = [row for row in source_rows if row.get("sample_usable") is True]
        valid = [row for row in source_rows if row.get("annotation_valid") is True]
        if availability == "error" and len(usable) != len(source_rows):
            raise RuntimeError(
                f"Manifest has {len(source_rows) - len(usable)} unusable of {len(source_rows)} rows. "
                "Use availability='available' only for an explicitly partial experiment."
            )
        if availability == "valid":
            valid_without_video = [row for row in valid if row.get("sample_usable") is not True]
            if valid_without_video:
                raise RuntimeError(
                    f"Manifest has {len(valid_without_video)} valid annotations without video. "
                    "Use availability='available' only for an explicitly partial experiment."
                )
            rows = valid
        elif availability == "available":
            rows = usable
        elif availability in ("all", "error"):
            rows = source_rows
        else:
            raise ValueError(f"Unknown availability policy: {availability}")
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            rows = rows[:limit]
        if not rows:
            raise ValueError("Dataset is empty")
        self.rows = rows
        self.num_frames = num_frames
        self.output_size = output_size
        self.decode_video = decode_video
        self.coverage = {
            "manifest_rows": len(source_rows),
            "usable_rows": len(usable),
            "valid_annotation_rows": len(valid),
            "selected_rows": len(rows),
            "availability_policy": availability,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def metadata(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "take_uid": row["take_uid"],
            "video_path": row["video"]["path"],
            "start_sec": row["video"]["clip_start"],
            "end_sec": row["video"]["clip_end"],
            "label": row["action"]["class_index"],
            "raw_class": row["action"]["label"],
            "action_name": row["action"]["name"],
            "sample_usable": row.get("sample_usable"),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.metadata(index)
        result = {"label": metadata["label"], "metadata": metadata}
        if self.decode_video:
            video, decode = read_video_clip(
                metadata["video_path"],
                float(metadata["start_sec"]),
                float(metadata["end_sec"]),
                num_frames=self.num_frames,
                output_size=self.output_size,
            )
            result.update(video=video, decode=decode)
        return result


def collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    result: dict[str, Any] = {
        "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "metadata": [sample["metadata"] for sample in samples],
    }
    if "video" in samples[0]:
        result["video"] = torch.stack([sample["video"] for sample in samples])
        result["decode"] = [sample["decode"] for sample in samples]
    return result
