"""Batch command-line interface for the face-blurring pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from oopsie_backend.face_blur.benchmark import load_manifest, parse_input_size
from oopsie_backend.face_blur.detectors import create_detector
from oopsie_backend.face_blur.models import (
    MODEL_SPECS,
    default_model_dir,
    model_metadata,
    provision_model,
)
from oopsie_backend.face_blur.pipeline import (
    BlurPipelineConfig,
    EncoderConfig,
    blur_video,
    utc_now,
)
from oopsie_backend.face_blur.render import BlurConfig
from oopsie_backend.face_blur.tracking import TrackingConfig

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
TRACKED_DETECTORS = tuple(
    name for name in MODEL_SPECS if name != "deface-centerface"
)


@dataclass(frozen=True, slots=True)
class VideoJob:
    job_id: str
    source_path: Path
    output_path: Path
    label: str | None = None


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


def _safe_output_path(output_dir: Path, relative_path: Path) -> Path:
    output = (output_dir / relative_path.with_suffix(".mp4")).resolve()
    if not output.is_relative_to(output_dir):
        raise ValueError(f"output path escapes output directory: {relative_path}")
    return output


def discover_jobs(
    *,
    output_dir: Path,
    inputs: list[Path] | None = None,
    manifest: Path | None = None,
    labels: set[str] | None = None,
) -> list[VideoJob]:
    """Resolve file, directory, or benchmark-manifest inputs into video jobs."""
    output_dir = output_dir.resolve()
    jobs: list[VideoJob] = []
    if manifest is not None:
        selected_labels = labels or {"yes", "no", "uncertain"}
        for sample in load_manifest(manifest):
            if sample.face_present not in selected_labels:
                continue
            jobs.append(
                VideoJob(
                    job_id=sample.sample_id,
                    source_path=sample.video_path.resolve(),
                    output_path=_safe_output_path(
                        output_dir, Path(sample.local_video_path)
                    ),
                    label=sample.face_present,
                )
            )
    else:
        for input_path in inputs or []:
            source = input_path.resolve()
            if source.is_file():
                if source.suffix.lower() not in VIDEO_EXTENSIONS:
                    raise ValueError(f"unsupported video extension: {source}")
                jobs.append(
                    VideoJob(
                        job_id=source.stem,
                        source_path=source,
                        output_path=_safe_output_path(output_dir, Path(source.name)),
                    )
                )
                continue
            if not source.is_dir():
                raise FileNotFoundError(f"input not found: {source}")
            for video_path in sorted(source.rglob("*")):
                if not video_path.is_file():
                    continue
                if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                resolved_video = video_path.resolve()
                if resolved_video.is_relative_to(output_dir):
                    continue
                relative_path = video_path.relative_to(source)
                jobs.append(
                    VideoJob(
                        job_id=relative_path.as_posix(),
                        source_path=resolved_video,
                        output_path=_safe_output_path(output_dir, relative_path),
                    )
                )

    outputs: dict[Path, Path] = {}
    for job in jobs:
        previous = outputs.get(job.output_path)
        if previous is not None and previous != job.source_path:
            raise ValueError(
                f"multiple inputs map to {job.output_path}: {previous} and "
                f"{job.source_path}"
            )
        outputs[job.output_path] = job.source_path
    return jobs


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import onnxruntime as ort
    except ImportError:
        return "cpu"
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "cuda:0"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Video file or directory; repeat for multiple roots.",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Face-blur test-set manifest or a compatible manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=["yes", "no", "uncertain"],
        help="Manifest labels to process (default: all).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--detector", choices=TRACKED_DETECTORS, default="scrfd-10g"
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:<index> (default: auto).",
    )
    parser.add_argument("--input-size", type=parse_input_size, default=(640, 640))
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.4)

    parser.add_argument("--max-gap-frames", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--pre-roll-frames", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--max-center-distance", type=float, default=1.75)
    parser.add_argument("--max-size-ratio", type=float, default=3.0)
    parser.add_argument("--motion-decay", type=float, default=0.9)
    parser.add_argument("--hold-expansion-per-frame", type=float, default=0.04)
    parser.add_argument("--padding-horizontal", type=float, default=0.25)
    parser.add_argument("--padding-top", type=float, default=0.35)
    parser.add_argument("--padding-bottom", type=float, default=0.25)

    parser.add_argument(
        "--blur-method", choices=["gaussian", "pixelate"], default="gaussian"
    )
    parser.add_argument("--gaussian-kernel-fraction", type=float, default=0.5)
    parser.add_argument("--gaussian-min-kernel", type=int, default=31)
    parser.add_argument("--gaussian-passes", type=int, default=2)
    parser.add_argument("--pixel-block-size", type=int, default=12)

    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--audio", choices=["copy", "aac", "drop"], default="copy")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _pipeline_config(args: argparse.Namespace) -> BlurPipelineConfig:
    tracking = TrackingConfig(
        max_gap_frames=args.max_gap_frames,
        hold_frames=args.hold_frames,
        pre_roll_frames=args.pre_roll_frames,
        smoothing_window=args.smoothing_window,
        min_iou=args.min_iou,
        max_center_distance=args.max_center_distance,
        max_size_ratio=args.max_size_ratio,
        motion_decay=args.motion_decay,
        hold_expansion_per_frame=args.hold_expansion_per_frame,
        padding_horizontal=args.padding_horizontal,
        padding_top=args.padding_top,
        padding_bottom=args.padding_bottom,
    )
    blur = BlurConfig(
        method=args.blur_method,
        gaussian_kernel_fraction=args.gaussian_kernel_fraction,
        gaussian_min_kernel=args.gaussian_min_kernel,
        gaussian_passes=args.gaussian_passes,
        pixel_block_size=args.pixel_block_size,
    )
    encoder = EncoderConfig(crf=args.crf, preset=args.preset, audio=args.audio)
    return BlurPipelineConfig(
        detector_name=args.detector,
        input_size=args.input_size,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        tracking=tracking,
        blur=blur,
        encoder=encoder,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.labels and args.manifest is None:
        parser.error("--labels can only be used with --manifest")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    output_dir = args.output_dir.resolve()
    try:
        jobs = discover_jobs(
            output_dir=output_dir,
            inputs=args.input,
            manifest=args.manifest,
            labels=set(args.labels) if args.labels else None,
        )
        if args.limit is not None:
            jobs = jobs[: args.limit]
        if not jobs:
            raise ValueError("no matching videos found")
        config = _pipeline_config(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        for job in jobs:
            print(f"{job.source_path} -> {job.output_path}")
        print(f"{len(jobs)} video(s)")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model_path = provision_model(args.detector, args.model_dir)
    model_info = model_metadata(args.detector, model_path)
    detector = create_detector(
        args.detector,
        model_path,
        input_size=config.input_size,
        score_threshold=config.score_threshold,
        nms_threshold=config.nms_threshold,
        device=device,
    )
    detector.detect(
        np.zeros((config.input_size[1], config.input_size[0], 3), dtype=np.uint8)
    )

    started = time.perf_counter()
    results: list[dict] = []
    failures = 0
    for job in tqdm(jobs, desc="face blur", unit="video", file=sys.stderr):
        try:
            result = blur_video(
                job.source_path,
                job.output_path,
                detector,
                config=config,
                detector_metadata=detector.metadata,
                model_metadata=model_info,
                resume=not args.no_resume,
                overwrite=args.overwrite,
            )
            results.append(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "source_path": str(job.source_path),
                    "output_path": str(job.output_path),
                    "status": result["status"],
                    "track_count": result.get("tracking", {}).get("track_count"),
                    "masked_frame_count": result.get("tracking", {}).get(
                        "masked_frame_count"
                    ),
                }
            )
        except Exception as exc:
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"{job.job_id}: {error}", file=sys.stderr)
            results.append(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "source_path": str(job.source_path),
                    "output_path": str(job.output_path),
                    "status": "error",
                    "error": error,
                }
            )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "device": device,
        "config": config.to_dict(),
        "model": model_info,
        "totals": {
            "videos": len(jobs),
            "successful": len(jobs) - failures,
            "failed": failures,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "results": results,
    }
    report_path = output_dir / "face-blur-run.json"
    _atomic_json(report_path, report)
    print(
        f"processed {len(jobs) - failures}/{len(jobs)} video(s); report: {report_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
