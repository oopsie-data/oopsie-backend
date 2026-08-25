"""Batch CenterFace anonymization with confirmed temporal tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from oopsie_backend.face_blur.cli import discover_jobs
from oopsie_backend.face_blur.detectors.deface import DefaceCenterFaceDetector
from oopsie_backend.face_blur.models import (
    default_model_dir,
    model_metadata,
    provision_model,
    sha256_file,
)
from oopsie_backend.face_blur.pipeline import (
    EncoderConfig,
    detect_video,
    render_video,
    utc_now,
)
from oopsie_backend.face_blur.render import (
    DefaceMaskConfig,
    blur_frame_deface,
)
from oopsie_backend.face_blur.tracking import (
    TrackConfirmationConfig,
    TrackingConfig,
    plan_confirmed_blur_regions,
)

SCHEMA_VERSION = 1
ALGORITHM_VERSION = 3


def _default_tracking() -> TrackingConfig:
    return TrackingConfig(
        max_gap_frames=8,
        hold_frames=2,
        pre_roll_frames=2,
        smoothing_window=1,
        min_iou=0.15,
        max_center_distance=1.0,
        max_size_ratio=2.0,
        motion_decay=0.9,
        hold_expansion_per_frame=0.02,
        padding_horizontal=0.0,
        padding_top=0.0,
        padding_bottom=0.0,
    )


@dataclass(frozen=True, slots=True)
class TrackedDefaceConfig:
    """Complete, auditable configuration for the tracked deface pipeline."""

    confirmation: TrackConfirmationConfig = field(
        default_factory=TrackConfirmationConfig
    )
    tracking: TrackingConfig = field(default_factory=_default_tracking)
    mask: DefaceMaskConfig = field(default_factory=DefaceMaskConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def to_dict(self) -> dict:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "detector_name": "deface-centerface",
            "input_size": None,
            "resolution_mode": "native",
            "temporal_tracking": True,
            "confirmation": self.confirmation.to_dict(),
            "tracking": self.tracking.to_dict(),
            "mask": self.mask.to_dict(),
            "encoder": self.encoder.to_dict(),
        }


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


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.tracked-deface.json")


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


def run_tracked_deface_video(
    source_path: Path,
    output_path: Path,
    *,
    detector: DefaceCenterFaceDetector,
    config: TrackedDefaceConfig,
    model_info: dict,
    resume: bool = True,
    overwrite: bool = False,
) -> dict:
    """Detect, confirm tracks, and render one video through deface masks."""
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
    detector_info = detector.metadata
    hash_config = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": config.to_dict(),
        "model_sha256": model_info["model_sha256"],
        "package_version": detector_info["package_version"],
        "providers": detector_info["providers"],
    }
    config_hash = _canonical_hash(hash_config)
    if resume and metadata_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
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
        score_threshold=config.confirmation.continuation_threshold,
    )
    planning_started = time.perf_counter()
    plan = plan_confirmed_blur_regions(
        detection_pass.detections_by_frame,
        frame_width=detection_pass.video.width,
        frame_height=detection_pass.video.height,
        tracking=config.tracking,
        confirmation=config.confirmation,
    )
    planning_seconds = time.perf_counter() - planning_started
    rendering_seconds = render_video(
        source_path,
        output_path,
        video=detection_pass.video,
        plan=plan,
        encoder_config=config.encoder,
        frame_renderer=partial(blur_frame_deface, config=config.mask),
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
        "detector": detector_info,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, action="append")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", choices=["yes", "no", "uncertain"])
    parser.add_argument(
        "--blur-actions", nargs="+", choices=["blur", "preserve", "review"]
    )
    parser.add_argument("--allow-incomplete-annotations", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--device", default="auto")

    parser.add_argument("--continuation-threshold", type=float, default=0.2)
    parser.add_argument("--seed-threshold", type=float, default=0.3)
    parser.add_argument("--min-seed-observations", type=int, default=2)
    parser.add_argument("--drop-unconfirmed-seeds", action="store_true")
    parser.add_argument("--fallback-pre-roll-frames", type=int, default=2)
    parser.add_argument("--fallback-hold-frames", type=int, default=15)
    parser.add_argument("--max-gap-frames", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--pre-roll-frames", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--max-center-distance", type=float, default=1.0)
    parser.add_argument("--max-size-ratio", type=float, default=2.0)
    parser.add_argument("--motion-decay", type=float, default=0.9)
    parser.add_argument("--hold-expansion-per-frame", type=float, default=0.02)

    parser.add_argument("--mask-scale", type=float, default=1.3)
    parser.add_argument("--boxes", action="store_true")
    parser.add_argument(
        "--replacewith", choices=["blur", "solid", "mosaic"], default="blur"
    )
    parser.add_argument("--mosaic-size", type=int, default=20)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--audio", choices=["copy", "aac", "drop"], default="copy")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> TrackedDefaceConfig:
    return TrackedDefaceConfig(
        confirmation=TrackConfirmationConfig(
            continuation_threshold=args.continuation_threshold,
            seed_threshold=args.seed_threshold,
            min_seed_observations=args.min_seed_observations,
            render_unconfirmed_seeds=not args.drop_unconfirmed_seeds,
            fallback_pre_roll_frames=args.fallback_pre_roll_frames,
            fallback_hold_frames=args.fallback_hold_frames,
        ),
        tracking=TrackingConfig(
            max_gap_frames=args.max_gap_frames,
            hold_frames=args.hold_frames,
            pre_roll_frames=args.pre_roll_frames,
            smoothing_window=args.smoothing_window,
            min_iou=args.min_iou,
            max_center_distance=args.max_center_distance,
            max_size_ratio=args.max_size_ratio,
            motion_decay=args.motion_decay,
            hold_expansion_per_frame=args.hold_expansion_per_frame,
            padding_horizontal=0.0,
            padding_top=0.0,
            padding_bottom=0.0,
        ),
        mask=DefaceMaskConfig(
            mask_scale=args.mask_scale,
            ellipse=not args.boxes,
            replacewith=args.replacewith,
            mosaic_size=args.mosaic_size,
        ),
        encoder=EncoderConfig(
            crf=args.crf,
            preset=args.preset,
            audio=args.audio,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.labels and args.manifest is None:
        parser.error("--labels can only be used with --manifest")
    if args.blur_actions and args.manifest is None:
        parser.error("--blur-actions can only be used with --manifest")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    output_dir = args.output_dir.resolve()
    try:
        jobs = discover_jobs(
            output_dir=output_dir,
            inputs=args.input,
            manifest=args.manifest,
            labels=set(args.labels) if args.labels else None,
            blur_actions=set(args.blur_actions) if args.blur_actions else None,
            require_complete_actions=(
                bool(args.blur_actions) and not args.allow_incomplete_annotations
            ),
        )
        if args.limit is not None:
            jobs = jobs[: args.limit]
        if not jobs:
            raise ValueError("no matching videos found")
        config = _config_from_args(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        for job in jobs:
            print(f"{job.source_path} -> {job.output_path}")
        print(f"{len(jobs)} video(s)")
        return 0

    try:
        model_path = provision_model("deface-centerface", args.model_dir)
        model_info = model_metadata("deface-centerface", model_path)
        detector = DefaceCenterFaceDetector(
            model_path,
            score_threshold=config.confirmation.continuation_threshold,
            device=args.device,
        )
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict] = []
    failures = 0
    for job in tqdm(jobs, desc="tracked deface", unit="video"):
        try:
            result = run_tracked_deface_video(
                job.source_path,
                job.output_path,
                detector=detector,
                config=config,
                model_info=model_info,
                resume=not args.no_resume,
                overwrite=args.overwrite,
            )
            policy = result.get("tracking", {}).get("policy", {})
            results.append(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "blur_action": job.blur_action,
                    "source_path": str(job.source_path),
                    "output_path": str(job.output_path),
                    "status": result["status"],
                    "track_count": result.get("tracking", {}).get("track_count"),
                    "confirmed_track_count": policy.get("confirmed_track_count"),
                    "fallback_detection_count": policy.get("fallback_detection_count"),
                    "masked_frame_count": result.get("tracking", {}).get(
                        "masked_frame_count"
                    ),
                }
            )
        except Exception as exc:
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"{job.job_id}: {error}")
            results.append(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "blur_action": job.blur_action,
                    "source_path": str(job.source_path),
                    "output_path": str(job.output_path),
                    "status": "error",
                    "error": error,
                }
            )

    detector_info = detector.metadata
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "runtime": {
            "package": detector_info["package"],
            "package_version": detector_info["package_version"],
            "providers": detector_info["providers"],
        },
        "model": model_info,
        "totals": {
            "videos": len(jobs),
            "successful": len(jobs) - failures,
            "failed": failures,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "results": results,
    }
    report_path = output_dir / "tracked-deface-run.json"
    _atomic_json(report_path, report)
    print(
        f"processed {len(jobs) - failures}/{len(jobs)} video(s); report: {report_path}"
    )
    return 1 if failures else 0


__all__ = [
    "TrackedDefaceConfig",
    "main",
    "metadata_path_for",
    "run_tracked_deface_video",
]
