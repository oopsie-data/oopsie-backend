"""Map an episode's cameras to the recipe's camera roles and write one video per role.

``cameras.roles`` maps each role to glob patterns over camera names, tried in
order; the first role to match a camera takes it, so a camera fills at most one
role. ``cameras.fallback`` fills a role that matched nothing with another role's
camera. A role that is still empty skips the episode (``missing: skip``) or is
left blank (``missing: blank``), which the writers turn into black frames.

Each role's video is resampled to one frame per unified step, resized to
``image_size`` and encoded with the ffmpeg of ``imageio-ffmpeg`` (H.264, ``crf``). A video that ends early
repeats its last frame.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from oopsie_backend.training.source import Skip

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def assign_roles(cameras: list[str], cfg: dict) -> dict[str, str | None]:
    """Camera name per role, or None for a blank role."""
    taken: set[str] = set()
    out: dict[str, str | None] = {}
    for role, patterns in cfg["roles"].items():
        out[role] = None
        for pattern in [patterns] if isinstance(patterns, str) else patterns:
            matches = sorted(c for c in cameras if c not in taken and fnmatch.fnmatch(c.lower(), pattern.lower()))
            if matches:
                out[role] = matches[0]
                taken.add(matches[0])
                break
    for role, other in cfg["fallback"].items():
        if out[role] is None:
            out[role] = out[other]
    empty = [r for r, c in out.items() if c is None]
    if empty and cfg["missing"] == "skip":
        raise Skip(f"no camera for role(s) {empty} among {sorted(cameras)}")
    return out


def _resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    if frame.shape[:2] == (height, width):
        return frame
    shrink = frame.shape[0] >= height and frame.shape[1] >= width
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def transcode(
    path: Path, source_steps: int, steps: np.ndarray, out: Path, size: tuple[int, int], fps: float, crf: int
) -> None:
    """Write the frames of ``path`` at the source steps ``steps`` (nondecreasing) to ``out``.

    Video frames are spread evenly over the source steps; frames are decoded once
    and streamed to ffmpeg.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise Skip(f"cannot open {path.name}")
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        capture.release()
        raise Skip(f"no frames in {path.name}")
    wanted = np.minimum((steps * num_frames) // source_steps, num_frames - 1)
    height, width = size
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-g", "2", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    position, frame = -1, None
    try:
        for index in wanted:
            while position < index:
                ok, decoded = capture.read()
                if not ok:
                    break
                position, frame = position + 1, decoded
            if frame is None:
                raise Skip(f"no frames in {path.name}")
            proc.stdin.write(np.ascontiguousarray(_resize(frame, size)).tobytes())
        proc.stdin.close()
    except BaseException:
        proc.kill()
        raise
    finally:
        capture.release()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}: {proc.stderr.read().decode(errors='replace')}")
