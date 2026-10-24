"""Shared utilities for dataset-to-HDF5 conversion scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

SCHEMA_VERSION = "oopsiedata_format_v1"
MAX_DIM = 1080  # must match validate.py MAX_IMAGE_SIZE


def _resize_frames(frames: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Resize (T, H, W, 3) frames so that max(H, W) <= max_dim."""
    h, w = frames.shape[1:3]
    if w <= max_dim and h <= max_dim:
        return frames
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    )


def _parse_fps(control_freq: str, default_fps: float = 15.0) -> float:
    try:
        parsed = float(control_freq)
    except (TypeError, ValueError):
        return default_fps
    return parsed if parsed > 0 else default_fps


def _decode_text(value: Any) -> str:
    """Decode bytes, numpy scalars/arrays, or arbitrary values to a plain str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if value.size == 0:
            return ""
        return _decode_text(value.reshape(-1)[0])
    return str(value)


def write_episode_annotations(
    file_handle: h5py.File,
    episode_id: str,
    success: float,
    lab_id: str = "",
    operator_name: str = "",
    policy_id: str = "",
    robot_id: str = "",
    control_freq: str = "",
) -> None:
    """Create and populate the episode_annotations group in *file_handle*."""
    group = file_handle.create_group("episode_annotations")
    group.attrs["episode_id"] = episode_id
    group.attrs["success"] = np.float32(success)
    group.attrs["failure_category"] = ""
    group.attrs["failure_description"] = ""
    group.attrs["would_retry"] = np.float32(0.0)
    group.attrs["lab_id"] = lab_id
    group.attrs["operator_name"] = operator_name
    group.attrs["policy_id"] = policy_id
    group.attrs["robot_id"] = robot_id
    group.attrs["control_freq"] = control_freq
