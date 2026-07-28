"""Evaluate tracked-deface planning from saved low-threshold detections."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from oopsie_backend.face_blur.benchmark import wilson_interval
from oopsie_backend.face_blur.pipeline import utc_now
from oopsie_backend.face_blur.tracked_deface_cli import TrackedDefaceConfig
from oopsie_backend.face_blur.tracking import plan_confirmed_blur_regions
from oopsie_backend.face_blur.types import FaceDetection

SCHEMA_VERSION = 1


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


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _load_result(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "ok":
        raise ValueError(f"saved detection is not successful: {path}")
    return result


def _detections_by_frame(result: dict) -> list[list[FaceDetection]]:
    frame_count = int(result["video"]["decoded_frame_count"])
    detections: list[list[FaceDetection]] = [[] for _ in range(frame_count)]
    for frame in result.get("detection_frames", []):
        frame_index = int(frame["frame_index"])
        detections[frame_index] = [
            FaceDetection(
                *[float(value) for value in item["bbox"]],
                float(item["score"]),
            )
            for item in frame.get("detections", [])
        ]
    return detections


def evaluate_saved_detections(
    detection_paths: list[Path],
    *,
    config: TrackedDefaceConfig | None = None,
) -> dict:
    """Return video and frame trigger metrics for a tracked-deface policy."""
    pipeline_config = config or TrackedDefaceConfig()
    started = time.perf_counter()
    videos: list[dict] = []
    counts = Counter()
    aggregate_sources = Counter()
    negative_planned_box_area = 0.0
    negative_estimated_mask_area = 0.0
    negative_pixel_frame_total = 0

    for path in sorted(detection_paths):
        result = _load_result(path)
        video = result["video"]
        detections = _detections_by_frame(result)
        width = int(video["width"])
        height = int(video["height"])
        frame_count = len(detections)
        plan = plan_confirmed_blur_regions(
            detections,
            frame_width=width,
            frame_height=height,
            tracking=pipeline_config.tracking,
            confirmation=pipeline_config.confirmation,
        )
        label = result["sample"]["face_present"]
        triggered = plan.masked_frame_count > 0
        counts[f"{label}_videos"] += 1
        counts[f"{label}_triggered"] += int(triggered)
        counts[f"{label}_frames"] += frame_count
        counts[f"{label}_masked_frames"] += plan.masked_frame_count
        counts[f"{label}_regions"] += plan.region_count
        aggregate_sources.update(plan.source_counts)

        if label == "no":
            negative_pixel_frame_total += frame_count * width * height
            for faces in plan.faces_by_frame.values():
                for face in faces:
                    negative_planned_box_area += face.box.area
                    scaled = face.box.scaled_about_center(
                        pipeline_config.mask.mask_scale
                    ).clipped(width, height)
                    shape_fraction = math.pi / 4 if pipeline_config.mask.ellipse else 1
                    negative_estimated_mask_area += scaled.area * shape_fraction

        videos.append(
            {
                "sample_id": result["sample"]["sample_id"],
                "face_present": label,
                "decoded_frames": frame_count,
                "triggered": triggered,
                "masked_frame_count": plan.masked_frame_count,
                "region_count": plan.region_count,
                "source_counts": dict(sorted(plan.source_counts.items())),
                "policy": plan.policy,
            }
        )

    true_positive = counts["yes_triggered"]
    false_negative = counts["yes_videos"] - true_positive
    false_positive = counts["no_triggered"]
    true_negative = counts["no_videos"] - false_positive
    uncertain_detected = counts["uncertain_triggered"]
    uncertain_not_detected = counts["uncertain_videos"] - uncertain_detected
    recall_interval = wilson_interval(true_positive, counts["yes_videos"])
    specificity_interval = wilson_interval(true_negative, counts["no_videos"])
    metrics = {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "uncertain_detected": uncertain_detected,
        "uncertain_not_detected": uncertain_not_detected,
        "recall": _ratio(true_positive, counts["yes_videos"]),
        "recall_wilson_95_low": recall_interval[0],
        "recall_wilson_95_high": recall_interval[1],
        "specificity": _ratio(true_negative, counts["no_videos"]),
        "specificity_wilson_95_low": specificity_interval[0],
        "specificity_wilson_95_high": specificity_interval[1],
        "precision": _ratio(true_positive, true_positive + false_positive),
        "negative_video_trigger_rate": _ratio(false_positive, counts["no_videos"]),
        "negative_frame_count": counts["no_frames"],
        "negative_masked_frame_count": counts["no_masked_frames"],
        "negative_masked_frame_rate": _ratio(
            counts["no_masked_frames"], counts["no_frames"]
        ),
        "negative_region_count": counts["no_regions"],
        "negative_planned_box_area_pixels": negative_planned_box_area,
        "negative_planned_box_area_fraction": _ratio(
            negative_planned_box_area, negative_pixel_frame_total
        ),
        "negative_estimated_mask_area_pixels": negative_estimated_mask_area,
        "negative_estimated_mask_area_fraction": _ratio(
            negative_estimated_mask_area, negative_pixel_frame_total
        ),
        "positive_frame_count": counts["yes_frames"],
        "positive_masked_frame_count": counts["yes_masked_frames"],
        "positive_masked_frame_rate": _ratio(
            counts["yes_masked_frames"], counts["yes_frames"]
        ),
        "uncertain_detection_rate": _ratio(
            uncertain_detected, counts["uncertain_videos"]
        ),
        "source_counts": dict(sorted(aggregate_sources.items())),
        "successful_videos": len(videos),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": pipeline_config.to_dict(),
        "totals": {
            "videos": len(videos),
            "decoded_frames": sum(int(video["decoded_frames"]) for video in videos),
            "planning_seconds": time.perf_counter() - started,
        },
        "metrics": metrics,
        "videos": videos,
        "notes": {
            "positive_masked_frame_rate": (
                "Fraction of all frames in positive-labeled videos with at least one "
                "planned mask; this is not face-frame recall."
            ),
            "negative_estimated_mask_area_fraction": (
                "Sum of clipped scaled ellipse areas divided by all negative-video "
                "pixels; overlapping masks are counted separately."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detection_paths = sorted(args.detections_dir.glob("*.json.gz"))
    if not detection_paths:
        raise SystemExit(f"no .json.gz detections found under {args.detections_dir}")
    report = evaluate_saved_detections(detection_paths)
    _atomic_json(args.output.resolve(), report)
    print(f"evaluated {len(detection_paths)} video(s); report: {args.output.resolve()}")
    return 0


__all__ = ["evaluate_saved_detections", "main"]
