"""Two-pass face detection, temporal planning, and video anonymization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oopsie_backend.face_blur.detectors.base import FaceDetector
from oopsie_backend.face_blur.models import sha256_file
from oopsie_backend.face_blur.render import BlurConfig, blur_frame
from oopsie_backend.face_blur.tracking import (
    TrackedFace,
    TrackingConfig,
    TrackingPlan,
    plan_blur_regions,
)
from oopsie_backend.face_blur.types import FaceDetection

SCHEMA_VERSION = 1
ALGORITHM_VERSION = 1


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """FFmpeg H.264 output settings."""

    crf: int = 18
    preset: str = "medium"
    audio: str = "copy"

    def __post_init__(self) -> None:
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if self.preset not in {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        }:
            raise ValueError("unsupported x264 preset")
        if self.audio not in {"copy", "aac", "drop"}:
            raise ValueError("audio must be 'copy', 'aac', or 'drop'")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlurPipelineConfig:
    detector_name: str = "scrfd-10g"
    input_size: tuple[int, int] = (640, 640)
    score_threshold: float = 0.3
    nms_threshold: float = 0.4
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be between zero and one")
        width, height = self.input_size
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("input dimensions must be positive multiples of 32")

    def to_dict(self) -> dict:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "detector_name": self.detector_name,
            "input_size": list(self.input_size),
            "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
            "tracking": self.tracking.to_dict(),
            "blur": self.blur.to_dict(),
            "encoder": self.encoder.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    expected_frame_count: int
    decoded_frame_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionPass:
    detections_by_frame: tuple[tuple[FaceDetection, ...], ...]
    video: VideoInfo
    inference_seconds: float
    elapsed_seconds: float


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _canonical_hash(data: dict) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.face-blur.json")


def detect_video(
    source_path: Path,
    detector: FaceDetector,
    *,
    score_threshold: float,
) -> DetectionPass:
    """Decode a video once and retain accepted detections for temporal planning."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            f"invalid video dimensions for {source_path}: {width}x{height}"
        )
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video frame rate for {source_path}: {fps}")

    detections_by_frame: list[tuple[FaceDetection, ...]] = []
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            actual_height, actual_width = frame.shape[:2]
            if not detections_by_frame:
                width, height = actual_width, actual_height
            elif (actual_width, actual_height) != (width, height):
                raise RuntimeError(
                    "video dimensions changed while decoding: "
                    f"expected {width}x{height}, got {actual_width}x{actual_height}"
                )
            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            inference_seconds += time.perf_counter() - inference_started
            accepted = tuple(
                detection
                for detection in detections
                if detection.score >= score_threshold and detection.area > 0
            )
            detections_by_frame.append(accepted)
    finally:
        capture.release()
    elapsed_seconds = time.perf_counter() - started
    if not detections_by_frame:
        raise RuntimeError(f"video decoded zero frames: {source_path}")
    if expected_frames > 0 and len(detections_by_frame) != expected_frames:
        raise RuntimeError(
            "video did not decode the advertised number of frames: "
            f"expected {expected_frames}, got {len(detections_by_frame)}"
        )
    return DetectionPass(
        detections_by_frame=tuple(detections_by_frame),
        video=VideoInfo(
            width=width,
            height=height,
            fps=fps,
            expected_frame_count=expected_frames,
            decoded_frame_count=len(detections_by_frame),
        ),
        inference_seconds=inference_seconds,
        elapsed_seconds=elapsed_seconds,
    )


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "video encoding requires imageio-ffmpeg; install a face-blur extra"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_command(
    *,
    executable: str,
    source_path: Path,
    output_path: Path,
    video: VideoInfo,
    config: EncoderConfig,
) -> list[str]:
    pixel_format = (
        "yuv420p" if video.width % 2 == 0 and video.height % 2 == 0 else "yuv444p"
    )
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{video.width}x{video.height}",
        "-r",
        f"{video.fps:.12g}",
        "-i",
        "pipe:0",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
    ]
    if config.audio != "drop":
        command.extend(["-map", "1:a?"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            pixel_format,
        ]
    )
    if config.audio == "copy":
        command.extend(["-c:a", "copy"])
    elif config.audio == "aac":
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-movflags", "+faststart", str(output_path)])
    return command


def validate_encoded_video(path: Path, expected: VideoInfo) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not produce a video: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not validate encoded video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"encoded video has no decodable frames: {path}")
    if (width, height) != (expected.width, expected.height):
        raise RuntimeError(
            "encoded dimensions changed: "
            f"expected {expected.width}x{expected.height}, got {width}x{height}"
        )
    if frame_count and frame_count != expected.decoded_frame_count:
        raise RuntimeError(
            "encoded frame count changed: "
            f"expected {expected.decoded_frame_count}, got {frame_count}"
        )


def render_video(
    source_path: Path,
    output_path: Path,
    *,
    video: VideoInfo,
    plan: TrackingPlan,
    encoder_config: EncoderConfig,
    blur_config: BlurConfig | None = None,
    frame_renderer: Callable[[np.ndarray, Sequence[TrackedFace]], np.ndarray]
    | None = None,
) -> float:
    """Render a completed temporal plan with the selected frame anonymizer."""
    if blur_config is not None and frame_renderer is not None:
        raise ValueError("provide blur_config or frame_renderer, not both")
    selected_blur = blur_config or BlurConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".mp4", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    capture: cv2.VideoCapture | None = None
    process: subprocess.Popen | None = None
    stderr_handle = tempfile.TemporaryFile()
    started = time.perf_counter()
    try:
        command = _ffmpeg_command(
            executable=_ffmpeg_executable(),
            source_path=source_path,
            output_path=temp_path,
            video=video,
            config=encoder_config,
        )
        process = subprocess.Popen(  # noqa: S603 - fixed executable and arguments
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
        )
        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not reopen video for rendering: {source_path}")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            faces = plan.faces_by_frame.get(frame_index, ())
            if not faces:
                output = frame
            elif frame_renderer is not None:
                output = frame_renderer(frame, faces)
            else:
                output = blur_frame(frame, faces, selected_blur)
            try:
                process.stdin.write(np.ascontiguousarray(output).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg stopped while receiving frames") from exc
            frame_index += 1
        if frame_index != video.decoded_frame_count:
            raise RuntimeError(
                "source frame count changed between passes: "
                f"detected {video.decoded_frame_count}, rendered {frame_index}"
            )
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            stderr_handle.seek(0)
            error = stderr_handle.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg exited with status {return_code}: {error}")
        validate_encoded_video(temp_path, video)
        os.replace(temp_path, output_path)
    except Exception:
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        if capture is not None:
            capture.release()
        stderr_handle.close()
        temp_path.unlink(missing_ok=True)
    return time.perf_counter() - started


def _load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _can_resume(
    metadata: dict,
    *,
    source_stat: dict[str, int],
    source_sha256: str,
    config_hash: str,
    output_path: Path,
) -> bool:
    return (
        metadata.get("status") == "ok"
        and metadata.get("config_hash") == config_hash
        and metadata.get("source", {}).get("size_bytes") == source_stat["size_bytes"]
        and metadata.get("source", {}).get("mtime_ns") == source_stat["mtime_ns"]
        and metadata.get("source", {}).get("sha256") == source_sha256
        and output_path.is_file()
        and output_path.stat().st_size == metadata.get("output", {}).get("size_bytes")
        and sha256_file(output_path) == metadata.get("output", {}).get("sha256")
    )


def blur_video(
    source_path: Path,
    output_path: Path,
    detector: FaceDetector,
    *,
    config: BlurPipelineConfig,
    detector_metadata: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    resume: bool = True,
    overwrite: bool = False,
) -> dict:
    """Anonymize one video and atomically publish it with an audit sidecar."""
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source video not found: {source_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output path must use the .mp4 extension")
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    metadata_path = metadata_path_for(output_path)
    source_stat = _source_stat(source_path)
    source_sha256 = sha256_file(source_path)
    model_info = model_metadata or {}
    hash_config = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": config.to_dict(),
        "model_sha256": model_info.get("model_sha256"),
    }
    config_hash = _canonical_hash(hash_config)
    if resume and metadata_path.is_file():
        try:
            cached = _load_metadata(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            cached = {}
        if _can_resume(
            cached,
            source_stat=source_stat,
            source_sha256=source_sha256,
            config_hash=config_hash,
            output_path=output_path,
        ):
            return {**cached, "status": "skipped"}
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists without matching resumable metadata: {output_path}; "
            "pass overwrite=True to replace it"
        )

    total_started = time.perf_counter()
    detection_pass = detect_video(
        source_path,
        detector,
        score_threshold=config.score_threshold,
    )
    plan_started = time.perf_counter()
    plan = plan_blur_regions(
        detection_pass.detections_by_frame,
        frame_width=detection_pass.video.width,
        frame_height=detection_pass.video.height,
        config=config.tracking,
    )
    planning_seconds = time.perf_counter() - plan_started
    rendering_seconds = render_video(
        source_path,
        output_path,
        video=detection_pass.video,
        plan=plan,
        encoder_config=config.encoder,
        blur_config=config.blur,
    )
    output_stat = output_path.stat()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "config_hash": config_hash,
        "source": {
            "path": str(source_path),
            **source_stat,
            "sha256": source_sha256,
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_stat.st_size,
            "sha256": sha256_file(output_path),
        },
        "video": detection_pass.video.to_dict(),
        "config": config.to_dict(),
        "detector": detector_metadata or detector.metadata,
        "model": model_info,
        "tracking": plan.to_summary_dict(),
        "performance": {
            "detection_pass_seconds": detection_pass.elapsed_seconds,
            "inference_seconds": detection_pass.inference_seconds,
            "planning_seconds": planning_seconds,
            "rendering_seconds": rendering_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
        "completed_at": utc_now(),
    }
    _atomic_json(metadata_path, result)
    return result


__all__ = [
    "BlurPipelineConfig",
    "DetectionPass",
    "EncoderConfig",
    "VideoInfo",
    "blur_video",
    "detect_video",
    "metadata_path_for",
    "render_video",
    "validate_encoded_video",
]
