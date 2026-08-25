"""Exhaustive video-level face detector benchmark.

The existing hand labels indicate whether a video contains any face-like human
content. This runner therefore reports video-level recall and false-positive
rates; frame-level privacy coverage requires the richer annotations planned for
the next evaluation phase.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from oopsie_backend.face_blur.detectors import FaceDetector, create_detector
from oopsie_backend.face_blur.models import (
    MODEL_SPECS,
    default_model_dir,
    model_metadata,
    provision_model,
    sha256_file,
)
from oopsie_backend.face_blur.types import FaceDetection

SCHEMA_VERSION = 1
ALLOWED_LABELS = frozenset({"yes", "no", "uncertain"})
ALLOWED_BLUR_ACTIONS = frozenset({"", "blur", "preserve", "review"})
ALLOWED_EXEMPTION_REASONS = frozenset(
    {"", "already_blurred", "task_relevant_depiction", "other"}
)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    sample_id: str
    source_kind: str
    source_repo: str
    source_path: str
    local_video_path: str
    video_path: Path
    face_present: str
    labeler: str
    notes: str
    blur_action: str = ""
    exemption_reason: str = ""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def parse_input_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("input size must look like 640x640") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("input dimensions must be positive")
    if width % 32 or height % 32:
        raise argparse.ArgumentTypeError("input dimensions must be multiples of 32")
    return width, height


def load_manifest(manifest_path: Path) -> list[BenchmarkSample]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    required = {
        "sample_id",
        "source_kind",
        "source_repo",
        "source_path",
        "local_video_path",
        "face_present",
        "labeler",
        "notes",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
    advanced_manifest = bool(rows and "blur_action" in rows[0])

    samples: list[BenchmarkSample] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, 2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"invalid or duplicate sample_id on row {row_number}")
        seen_ids.add(sample_id)
        label = (row.get("face_present") or "").strip().lower()
        if label not in ALLOWED_LABELS and not (advanced_manifest and not label):
            raise ValueError(
                f"invalid face_present={label!r} on row {row_number}; "
                f"expected one of {sorted(ALLOWED_LABELS)}"
            )
        blur_action = (row.get("blur_action") or "").strip().lower()
        if blur_action not in ALLOWED_BLUR_ACTIONS:
            raise ValueError(
                f"invalid blur_action={blur_action!r} on row {row_number}; "
                f"expected one of {sorted(ALLOWED_BLUR_ACTIONS)}"
            )
        exemption_reason = (row.get("exemption_reason") or "").strip().lower()
        if exemption_reason not in ALLOWED_EXEMPTION_REASONS:
            raise ValueError(
                f"invalid exemption_reason={exemption_reason!r} on row "
                f"{row_number}; expected one of "
                f"{sorted(ALLOWED_EXEMPTION_REASONS)}"
            )
        relative_path = Path((row.get("local_video_path") or "").strip())
        video_path = (root / relative_path).resolve()
        if not video_path.is_relative_to(root):
            raise ValueError(
                f"local_video_path escapes the test-set root on row {row_number}"
            )
        if not video_path.is_file():
            raise FileNotFoundError(
                f"video not found on row {row_number}: {video_path}"
            )
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                source_kind=(row.get("source_kind") or "").strip(),
                source_repo=(row.get("source_repo") or "").strip(),
                source_path=(row.get("source_path") or "").strip(),
                local_video_path=relative_path.as_posix(),
                video_path=video_path,
                face_present=label,
                labeler=(row.get("labeler") or "").strip(),
                notes=(row.get("notes") or "").strip(),
                blur_action=blur_action,
                exemption_reason=exemption_reason,
            )
        )
    if not samples:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    return samples


def _canonical_hash(data: dict) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_gzip(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _can_resume(
    result: dict,
    *,
    config_hash: str,
    source_stat: dict[str, int],
) -> bool:
    return (
        result.get("status") == "ok"
        and result.get("config_hash") == config_hash
        and result.get("source", {}).get("size_bytes") == source_stat["size_bytes"]
        and result.get("source", {}).get("mtime_ns") == source_stat["mtime_ns"]
    )


def _draw_review_frame(
    frame: np.ndarray,
    detections: Iterable[FaceDetection],
    *,
    threshold: float,
    detector_name: str,
    sample: BenchmarkSample,
    frame_index: int,
) -> np.ndarray:
    output = frame.copy()
    accepted = [detection for detection in detections if detection.score >= threshold]
    for detection in accepted:
        start = (int(round(detection.x1)), int(round(detection.y1)))
        end = (int(round(detection.x2)), int(round(detection.y2)))
        cv2.rectangle(output, start, end, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{detection.score:.3f}",
            (start[0], max(14, start[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    title = (
        f"{detector_name} | {sample.sample_id} | label={sample.face_present} | "
        f"frame={frame_index} | >= {threshold:g}: {len(accepted)}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def benchmark_video(
    detector: FaceDetector,
    detector_name: str,
    sample: BenchmarkSample,
    *,
    config_hash: str,
    max_frames: int | None,
    review_path: Path | None,
    review_threshold: float,
) -> dict:
    """Run one detector over one complete video and return serializable results."""
    source_stat = _source_stat(sample.video_path)
    source_hash = sha256_file(sample.video_path)
    capture = cv2.VideoCapture(str(sample.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {sample.video_path}")

    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    midpoint = max(0, expected_frames // 2)
    fallback_frame: np.ndarray | None = None
    fallback_index = 0
    top_frame: np.ndarray | None = None
    top_frame_index = 0
    top_frame_detections: list[FaceDetection] = []
    top_score = -1.0
    detection_frames: list[dict] = []
    frame_index = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if fallback_frame is None or frame_index == midpoint:
                fallback_frame = frame.copy()
                fallback_index = frame_index
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            inference_started = time.perf_counter()
            detections = detector.detect(frame)
            inference_seconds += time.perf_counter() - inference_started
            if detections:
                frame_score = max(detection.score for detection in detections)
                detection_frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": round(timestamp_ms, 3),
                        "detections": [detection.to_dict() for detection in detections],
                    }
                )
                if frame_score > top_score:
                    top_score = frame_score
                    top_frame = frame.copy()
                    top_frame_index = frame_index
                    top_frame_detections = detections
            frame_index += 1
    finally:
        capture.release()
    elapsed_seconds = time.perf_counter() - started
    if frame_index == 0:
        raise RuntimeError(f"video decoded zero frames: {sample.video_path}")

    if review_path is not None:
        review_source = top_frame if top_frame is not None else fallback_frame
        review_index = top_frame_index if top_frame is not None else fallback_index
        review_detections = top_frame_detections if top_frame is not None else []
        if review_source is not None:
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review = _draw_review_frame(
                review_source,
                review_detections,
                threshold=review_threshold,
                detector_name=detector_name,
                sample=sample,
                frame_index=review_index,
            )
            if not cv2.imwrite(str(review_path), review):
                raise RuntimeError(f"could not write review image: {review_path}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "config_hash": config_hash,
        "sample": {
            key: value for key, value in asdict(sample).items() if key != "video_path"
        },
        "source": {
            "path": str(sample.video_path),
            **source_stat,
            "sha256": source_hash,
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": codec,
            "expected_frame_count": expected_frames,
            "decoded_frame_count": frame_index,
            "partial": max_frames is not None and frame_index < expected_frames,
        },
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "inference_seconds": inference_seconds,
            "pipeline_fps": frame_index / elapsed_seconds if elapsed_seconds else None,
            "inference_fps": frame_index / inference_seconds
            if inference_seconds
            else None,
        },
        "detection_frame_count": len(detection_frames),
        "detection_count": sum(len(frame["detections"]) for frame in detection_frames),
        "max_score": max((top_score, 0.0)),
        "detection_frames": detection_frames,
        "completed_at": utc_now(),
    }


def _frame_max_scores(result: dict) -> list[float]:
    return [
        max(float(detection["score"]) for detection in frame["detections"])
        for frame in result.get("detection_frames", [])
        if frame.get("detections")
    ]


def max_consecutive_detection_frames(result: dict, threshold: float) -> int:
    """Longest run of adjacent decoded frames with a detection at threshold."""
    indices = [
        int(frame["frame_index"])
        for frame in result.get("detection_frames", [])
        if frame.get("detections")
        and max(float(detection["score"]) for detection in frame["detections"])
        >= threshold
    ]
    longest = 0
    current = 0
    previous: int | None = None
    for index in indices:
        current = current + 1 if previous is not None and index == previous + 1 else 1
        longest = max(longest, current)
        previous = index
    return longest


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (center - margin) / denominator, (center + margin) / denominator


def threshold_metrics(results: list[dict], threshold: float) -> dict:
    successful = [result for result in results if result.get("status") == "ok"]
    counts = {
        "true_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "false_positive": 0,
        "uncertain_detected": 0,
        "uncertain_not_detected": 0,
    }
    for result in successful:
        predicted = float(result.get("max_score", 0.0)) >= threshold
        label = result["sample"]["face_present"]
        if label == "yes":
            counts["true_positive" if predicted else "false_negative"] += 1
        elif label == "no":
            counts["false_positive" if predicted else "true_negative"] += 1
        else:
            counts["uncertain_detected" if predicted else "uncertain_not_detected"] += 1
    positive_total = counts["true_positive"] + counts["false_negative"]
    negative_total = counts["true_negative"] + counts["false_positive"]
    uncertain_total = counts["uncertain_detected"] + counts["uncertain_not_detected"]
    negative_frame_total = sum(
        int(result.get("video", {}).get("decoded_frame_count", 0))
        for result in successful
        if result["sample"]["face_present"] == "no"
    )
    negative_pixel_frame_total = sum(
        int(result.get("video", {}).get("decoded_frame_count", 0))
        * int(result.get("video", {}).get("width", 0))
        * int(result.get("video", {}).get("height", 0))
        for result in successful
        if result["sample"]["face_present"] == "no"
    )
    negative_triggered_frames = 0
    negative_detection_count = 0
    negative_raw_box_area_pixels = 0.0
    for result in successful:
        if result["sample"]["face_present"] != "no":
            continue
        for frame in result.get("detection_frames", []):
            accepted = [
                detection
                for detection in frame.get("detections", [])
                if float(detection["score"]) >= threshold
            ]
            if accepted:
                negative_triggered_frames += 1
                negative_detection_count += len(accepted)
                for detection in accepted:
                    x1, y1, x2, y2 = (float(value) for value in detection["bbox"])
                    negative_raw_box_area_pixels += max(0.0, x2 - x1) * max(
                        0.0, y2 - y1
                    )
    recall_interval = wilson_interval(counts["true_positive"], positive_total)
    specificity_interval = wilson_interval(counts["true_negative"], negative_total)
    return {
        "threshold": threshold,
        **counts,
        "recall": _ratio(counts["true_positive"], positive_total),
        "recall_wilson_95_low": recall_interval[0],
        "recall_wilson_95_high": recall_interval[1],
        "specificity": _ratio(counts["true_negative"], negative_total),
        "specificity_wilson_95_low": specificity_interval[0],
        "specificity_wilson_95_high": specificity_interval[1],
        "precision": _ratio(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        ),
        "negative_video_trigger_rate": _ratio(counts["false_positive"], negative_total),
        "negative_frame_count": negative_frame_total,
        "negative_triggered_frame_count": negative_triggered_frames,
        "negative_frame_trigger_rate": _ratio(
            negative_triggered_frames, negative_frame_total
        ),
        "negative_detection_count": negative_detection_count,
        "negative_detections_per_1000_frames": (
            1000 * negative_detection_count / negative_frame_total
            if negative_frame_total
            else None
        ),
        "negative_raw_box_area_pixels": negative_raw_box_area_pixels,
        "negative_raw_box_area_fraction": (
            negative_raw_box_area_pixels / negative_pixel_frame_total
            if negative_pixel_frame_total
            else None
        ),
        "uncertain_detection_rate": _ratio(
            counts["uncertain_detected"], uncertain_total
        ),
        "successful_videos": len(successful),
    }


def _aggregate_by_repo(results: list[dict], threshold: float) -> dict[str, dict]:
    repos: dict[str, list[dict]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        repo = result["sample"].get("source_repo") or "unknown"
        repos.setdefault(repo, []).append(result)
    return {
        repo: threshold_metrics(repo_results, threshold)
        for repo, repo_results in sorted(repos.items())
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_detector_report(
    output_dir: Path,
    detector_name: str,
    results: list[dict],
    *,
    thresholds: list[float],
    primary_threshold: float,
    detector_metadata: dict,
    model_info: dict,
    config: dict,
    warmup_seconds: float,
) -> dict:
    metrics = [threshold_metrics(results, threshold) for threshold in thresholds]
    successful = [result for result in results if result.get("status") == "ok"]
    errors = [result for result in results if result.get("status") != "ok"]
    total_frames = sum(
        int(result.get("video", {}).get("decoded_frame_count", 0))
        for result in successful
    )
    total_elapsed = sum(
        float(result.get("performance", {}).get("elapsed_seconds", 0.0))
        for result in successful
    )
    total_inference = sum(
        float(result.get("performance", {}).get("inference_seconds", 0.0))
        for result in successful
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "git_revision": git_revision(),
        "git_dirty": git_is_dirty(),
        "detector": detector_name,
        "detector_metadata": detector_metadata,
        "model": model_info,
        "config": config,
        "warmup_seconds": warmup_seconds,
        "totals": {
            "videos": len(results),
            "successful_videos": len(successful),
            "failed_videos": len(errors),
            "decoded_frames": total_frames,
            "elapsed_seconds": total_elapsed,
            "inference_seconds": total_inference,
            "pipeline_fps": total_frames / total_elapsed if total_elapsed else None,
            "inference_fps": total_frames / total_inference
            if total_inference
            else None,
        },
        "threshold_metrics": metrics,
        "primary_threshold": primary_threshold,
        "primary_metrics": threshold_metrics(results, primary_threshold),
        "primary_metrics_by_repo": _aggregate_by_repo(results, primary_threshold),
        "errors": [
            {
                "sample_id": result.get("sample", {}).get("sample_id"),
                "error": result.get("error"),
            }
            for result in errors
        ],
    }
    _atomic_json(output_dir / "summary.json", report)
    _write_csv(
        output_dir / "thresholds.csv",
        list(metrics[0].keys()),
        metrics,
    )

    video_rows: list[dict] = []
    threshold_field = f"frames_at_{str(primary_threshold).replace('.', '_')}"
    for result in results:
        sample = result.get("sample", {})
        scores = _frame_max_scores(result)
        video_rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "face_present": sample.get("face_present"),
                "source_repo": sample.get("source_repo"),
                "source_path": sample.get("source_path"),
                "status": result.get("status"),
                "error": result.get("error"),
                "decoded_frames": result.get("video", {}).get("decoded_frame_count"),
                "max_score": result.get("max_score"),
                "detection_frames_at_inference_floor": len(scores),
                threshold_field: sum(score >= primary_threshold for score in scores),
                "max_consecutive_frames_at_primary_threshold": (
                    max_consecutive_detection_frames(result, primary_threshold)
                ),
                "detection_count_at_inference_floor": result.get("detection_count"),
                "pipeline_fps": result.get("performance", {}).get("pipeline_fps"),
                "inference_fps": result.get("performance", {}).get("inference_fps"),
                "notes": sample.get("notes"),
            }
        )
    video_fields = list(video_rows[0].keys()) if video_rows else []
    _write_csv(output_dir / "videos.csv", video_fields, video_rows)
    return report


def run_detector_benchmark(
    detector_name: str,
    samples: list[BenchmarkSample],
    *,
    model_dir: Path,
    output_root: Path,
    input_size: tuple[int, int],
    inference_threshold: float,
    nms_threshold: float,
    thresholds: list[float],
    primary_threshold: float,
    device: str,
    max_frames: int | None,
    resume: bool,
    review_images: bool,
) -> dict:
    model_path = provision_model(detector_name, model_dir)
    model_info = model_metadata(detector_name, model_path)
    effective_input_size = (
        None if detector_name == "deface-centerface" else list(input_size)
    )
    effective_device = device if detector_name != "yunet" else "cpu"
    raw_config = {
        "schema_version": SCHEMA_VERSION,
        "detector": detector_name,
        "model_sha256": model_info["model_sha256"],
        "input_size": effective_input_size,
        "inference_threshold": inference_threshold,
        "nms_threshold": nms_threshold,
        "device": effective_device,
        "max_frames": max_frames,
    }
    config_hash = _canonical_hash(raw_config)
    config = {**raw_config, "config_hash": config_hash}
    detector_output = output_root / detector_name
    result_dir = detector_output / "detections"
    review_dir = detector_output / "review"
    _atomic_json(detector_output / "config.json", config)

    initialization_started = time.perf_counter()
    detector = create_detector(
        detector_name,
        model_path,
        input_size=input_size,
        score_threshold=inference_threshold,
        nms_threshold=nms_threshold,
        device=device,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    warmup_started = time.perf_counter()
    detector.detect(np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8))
    warmup_seconds = time.perf_counter() - warmup_started

    results: list[dict] = []
    for sample in tqdm(samples, desc=detector_name, unit="video", file=sys.stderr):
        result_path = result_dir / f"{sample.sample_id}.json.gz"
        source_stat = _source_stat(sample.video_path)
        if resume and result_path.is_file():
            try:
                cached = _load_gzip_json(result_path)
            except (OSError, ValueError, json.JSONDecodeError):
                cached = {}
            if _can_resume(
                cached,
                config_hash=config_hash,
                source_stat=source_stat,
            ):
                results.append(cached)
                continue
        review_path = None
        if review_images:
            review_path = (
                review_dir / sample.face_present / f"{sample.sample_id}__review.jpg"
            )
        try:
            result = benchmark_video(
                detector,
                detector_name,
                sample,
                config_hash=config_hash,
                max_frames=max_frames,
                review_path=review_path,
                review_threshold=primary_threshold,
            )
        except Exception as exc:
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "config_hash": config_hash,
                "sample": {
                    key: value
                    for key, value in asdict(sample).items()
                    if key != "video_path"
                },
                "source": {"path": str(sample.video_path), **source_stat},
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at": utc_now(),
            }
            tqdm.write(
                f"{detector_name} {sample.sample_id}: {result['error']}",
                file=sys.stderr,
            )
        _atomic_json_gzip(result_path, result)
        results.append(result)

    report = write_detector_report(
        detector_output,
        detector_name,
        results,
        thresholds=thresholds,
        primary_threshold=primary_threshold,
        detector_metadata={
            **detector.metadata,
            "initialization_seconds": initialization_seconds,
        },
        model_info=model_info,
        config=config,
        warmup_seconds=warmup_seconds,
    )
    return report


def _write_comparison(output_root: Path, reports: list[dict]) -> None:
    rows: list[dict] = []
    for report in reports:
        for metrics in report["threshold_metrics"]:
            rows.append(
                {
                    "detector": report["detector"],
                    **report["totals"],
                    **metrics,
                }
            )
    if rows:
        _write_csv(output_root / "comparison.csv", list(rows[0].keys()), rows)
    _atomic_json(
        output_root / "comparison.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "detectors": reports,
        },
    )


def compare_existing_reports(
    output_root: Path, detector_names: Iterable[str] | None = None
) -> dict:
    """Combine already-generated detector summaries without rerunning inference."""
    names = list(detector_names or MODEL_SPECS)
    reports: list[dict] = []
    for name in names:
        summary_path = output_root / name / "summary.json"
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"detector summary not found: {summary_path}"
            ) from exc
        if not isinstance(report, dict) or report.get("detector") != name:
            raise ValueError(f"invalid detector summary: {summary_path}")
        reports.append(report)
    _write_comparison(output_root, reports)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "detectors": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("face_blur_testset/manifest.csv"),
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=None,
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("face_blur_benchmarks/latest")
    )
    parser.add_argument("--input-size", type=parse_input_size, default=(640, 640))
    parser.add_argument("--inference-threshold", type=float, default=0.1)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument("--primary-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug only: stop after this many frames per video.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching per-video result files and recompute.",
    )
    parser.add_argument(
        "--no-review-images",
        action="store_true",
        help="Do not write one annotated JPEG per video and detector.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.inference_threshold <= 1:
        parser.error("--inference-threshold must be between 0 and 1")
    if not 0 <= args.nms_threshold <= 1:
        parser.error("--nms-threshold must be between 0 and 1")
    thresholds = sorted(set(args.thresholds + [args.primary_threshold]))
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        parser.error("all evaluation thresholds must be between 0 and 1")
    if args.inference_threshold > min(thresholds):
        parser.error("--inference-threshold must not exceed evaluation thresholds")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    try:
        samples = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.limit is not None:
        samples = samples[: args.limit]

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for detector_name in args.detectors or MODEL_SPECS:
        display_input = (
            "native"
            if detector_name == "deface-centerface"
            else f"{args.input_size[0]}x{args.input_size[1]}"
        )
        display_device = args.device if detector_name != "yunet" else "cpu"
        print(
            f"Benchmarking {detector_name}: {len(samples)} video(s), "
            f"input={display_input}, device={display_device}",
            file=sys.stderr,
        )
        try:
            report = run_detector_benchmark(
                detector_name,
                samples,
                model_dir=args.model_dir,
                output_root=output_root,
                input_size=args.input_size,
                inference_threshold=args.inference_threshold,
                nms_threshold=args.nms_threshold,
                thresholds=thresholds,
                primary_threshold=args.primary_threshold,
                device=args.device,
                max_frames=args.max_frames,
                resume=not args.no_resume,
                review_images=not args.no_review_images,
            )
        except Exception as exc:
            print(
                f"error: {detector_name}: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            return 1
        reports.append(report)
        primary = report["primary_metrics"]
        print(
            f"  threshold={args.primary_threshold:g}: "
            f"recall={primary['recall']}, specificity={primary['specificity']}, "
            f"errors={report['totals']['failed_videos']}",
            file=sys.stderr,
        )
    _write_comparison(output_root, reports)
    print(f"Benchmark report: {output_root / 'comparison.json'}", file=sys.stderr)
    return 1 if any(report["totals"]["failed_videos"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
