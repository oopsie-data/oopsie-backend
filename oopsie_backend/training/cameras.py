"""Assign camera roles and write resized, resampled H.264 videos.

Matches unified camera names to role patterns, applies fallbacks and missing-view
rules, and selects source video frames for each prepared step.
"""

from __future__ import annotations

import fnmatch
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from oopsie_backend.training.read_episode import Skip

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def assign_roles(cameras: dict[str, str], cfg: dict) -> dict[str, str | None]:
    """Source camera per role, or None for a blank role; ``cameras`` maps unified names to source cameras."""
    taken: set[str] = set()
    out: dict[str, str | None] = {}
    for role, patterns in cfg["roles"].items():
        out[role] = None
        for pattern in [patterns] if isinstance(patterns, str) else patterns:
            matches = sorted(n for n in cameras if n not in taken and fnmatch.fnmatch(n, pattern))
            if matches:
                out[role] = cameras[matches[0]]
                taken.add(matches[0])
                break
    for role, other in cfg["fallback"].items():
        if out[role] is None:
            out[role] = out[other]
    if all(c is None for c in out.values()):
        raise Skip(f"no camera for any role among {sorted(cameras)}")
    empty = [r for r, c in out.items() if c is None and r not in cfg["optional"]]
    if empty and cfg["missing"] == "skip":
        raise Skip(f"no camera for role(s) {empty} among {sorted(cameras)}")
    return out


def assign_episode_roles(
    unified: list[dict] | None, videos: dict[str, Path], cfg: dict,
) -> dict[str, str | None]:
    """Map the index's unified names to the recipe's slots for the available videos."""
    if unified is None:
        raise Skip("no unified camera names; run oopsie index camera-names")
    names = {e["name"]: e["camera"] for e in unified if e["name"] and e["camera"] in videos}
    if not names:
        raise Skip("no camera has a unified name")
    return assign_roles(names, cfg)


def _resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    if frame.shape[:2] == (height, width):
        return frame
    shrink = frame.shape[0] >= height and frame.shape[1] >= width
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


@contextmanager
def opencv_fallback_to_ffmpeg(path: Path) -> Iterator[Path | None]:
    """Yield a temporary H.264 copy via FFmpeg when OpenCV cannot decode a source, such as some AV1 clips."""
    with tempfile.TemporaryDirectory(prefix="oopsie-video-") as tmp:
        decoded = Path(tmp) / "decoded.mp4"
        result = subprocess.run(
            [
                FFMPEG, "-nostdin", "-loglevel", "error", "-xerror", "-i", str(path),
                "-map", "0:v:0", "-an", "-vsync", "0", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "0", "-pix_fmt", "yuv420p", str(decoded),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
        )
        yield decoded if result.returncode == 0 and decoded.is_file() else None


def transcode(
    path: Path, source_steps: int, steps: np.ndarray, out: Path, size: tuple[int, int] | None, fps: float, crf: int
) -> None:
    """Prepare a source camera, falling back if OpenCV cannot decode its first frame."""
    try:
        _transcode(path, source_steps, steps, out, size, fps, crf)
    except Skip:
        with opencv_fallback_to_ffmpeg(path) as decoded:
            if decoded is None:
                raise
            _transcode(decoded, source_steps, steps, out, size, fps, crf)


def _transcode(
    path: Path, source_steps: int, steps: np.ndarray, out: Path, size: tuple[int, int] | None, fps: float, crf: int
) -> None:
    """Write the frames of ``path`` at the source steps ``steps`` (nondecreasing) to ``out``.

    Video frames are spread evenly over the source steps; frames are decoded once
    and streamed to ffmpeg.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise Skip(f"cannot open {path.name}")
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        capture.release()
        raise Skip(f"no frames in {path.name}")
    wanted = np.minimum((steps * num_frames) // source_steps, num_frames - 1)
    if size is None:
        h, w = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        size = (h - h % 2, w - w % 2)
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
        proc.wait()
        raise
    finally:
        capture.release()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}: {proc.stderr.read().decode(errors='replace')}")
