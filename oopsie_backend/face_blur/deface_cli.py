"""Batch wrapper around the native ORB-HD/deface video pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from oopsie_backend.face_blur.cli import discover_jobs
from oopsie_backend.face_blur.detectors.deface import execution_provider_for_device
from oopsie_backend.face_blur.models import (
    default_model_dir,
    model_metadata,
    provision_model,
    sha256_file,
)
from oopsie_backend.face_blur.pipeline import (
    VideoInfo,
    utc_now,
    validate_encoded_video,
)

SCHEMA_VERSION = 1
ALGORITHM_VERSION = 1


@dataclass(frozen=True, slots=True)
class NativeDefaceConfig:
    threshold: float = 0.2
    mask_scale: float = 1.3
    ellipse: bool = True
    replacewith: str = "blur"
    mosaic_size: int = 20
    audio: str = "copy"

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if self.mask_scale <= 0:
            raise ValueError("mask_scale must be positive")
        if self.replacewith not in {"blur", "solid", "mosaic"}:
            raise ValueError("unsupported replacement method")
        if self.mosaic_size <= 0:
            raise ValueError("mosaic_size must be positive")
        if self.audio not in {"copy", "drop"}:
            raise ValueError("audio must be 'copy' or 'drop'")

    def to_dict(self) -> dict:
        return {
            "algorithm_version": ALGORITHM_VERSION,
            **asdict(self),
            "input_size": None,
            "resolution_mode": "native",
            "temporal_tracking": False,
            # imageio otherwise pads dimensions to a multiple of 16.  That
            # silently changed a 640x360 source into a 640x368 output.
            "ffmpeg_config": {"codec": "libx264", "macro_block_size": 1},
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
    return output_path.with_name(f"{output_path.name}.deface.json")


def _video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"invalid source video metadata: {width}x{height}, {fps} fps, "
            f"{frame_count} frames"
        )
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        expected_frame_count=frame_count,
        decoded_frame_count=frame_count,
    )


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


def run_native_deface_video(
    source_path: Path,
    output_path: Path,
    *,
    centerface: object,
    config: NativeDefaceConfig,
    package_version: str,
    provider: str,
    model_info: dict,
    resume: bool = True,
    overwrite: bool = False,
) -> dict:
    """Run deface's own per-frame masks and atomically publish one video."""
    from deface.deface import video_detect

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
    hash_config = {
        "schema_version": SCHEMA_VERSION,
        "package": "deface",
        "package_version": package_version,
        "model_sha256": model_info["model_sha256"],
        "provider": provider,
        "pipeline": config.to_dict(),
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

    video = _video_info(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".mp4", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        video_detect(
            ipath=str(source_path),
            opath=str(temp_path),
            centerface=centerface,
            threshold=config.threshold,
            enable_preview=False,
            cam=False,
            nested=False,
            replacewith=config.replacewith,
            mask_scale=config.mask_scale,
            ellipse=config.ellipse,
            draw_scores=False,
            ffmpeg_config={"codec": "libx264", "macro_block_size": 1},
            keep_audio=config.audio == "copy",
            mosaicsize=config.mosaic_size,
        )
        validate_encoded_video(temp_path, video)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)

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
        "video": video.to_dict(),
        "config": config.to_dict(),
        "runtime": {
            "package": "deface",
            "package_version": package_version,
            "provider": provider,
        },
        "model": model_info,
        "performance": {"total_seconds": time.perf_counter() - started},
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
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--mask-scale", type=float, default=1.3)
    parser.add_argument("--boxes", action="store_true")
    parser.add_argument(
        "--replacewith", choices=["blur", "solid", "mosaic"], default="blur"
    )
    parser.add_argument("--mosaic-size", type=int, default=20)
    parser.add_argument("--audio", choices=["copy", "drop"], default="copy")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


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
        config = NativeDefaceConfig(
            threshold=args.threshold,
            mask_scale=args.mask_scale,
            ellipse=not args.boxes,
            replacewith=args.replacewith,
            mosaic_size=args.mosaic_size,
            audio=args.audio,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        for job in jobs:
            print(f"{job.source_path} -> {job.output_path}")
        print(f"{len(jobs)} video(s)")
        return 0

    try:
        import deface
        import onnxruntime as ort
        from deface.centerface import CenterFace
    except ImportError as exc:
        parser.error(f"install the face-blur project extra: {exc}")
    provider = execution_provider_for_device(
        args.device, list(ort.get_available_providers())
    )
    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    model_path = provision_model("deface-centerface", args.model_dir)
    model_info = model_metadata("deface-centerface", model_path)
    centerface = CenterFace(
        onnx_path=str(model_path),
        in_shape=None,
        backend="onnxrt",
        override_execution_provider=provider,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict] = []
    failures = 0
    for job in tqdm(jobs, desc="native deface", unit="video"):
        try:
            result = run_native_deface_video(
                job.source_path,
                job.output_path,
                centerface=centerface,
                config=config,
                package_version=deface.__version__,
                provider=provider,
                model_info=model_info,
                resume=not args.no_resume,
                overwrite=args.overwrite,
            )
            results.append(
                {
                    "job_id": job.job_id,
                    "label": job.label,
                    "blur_action": job.blur_action,
                    "source_path": str(job.source_path),
                    "output_path": str(job.output_path),
                    "status": result["status"],
                    "elapsed_seconds": result.get("performance", {}).get(
                        "total_seconds"
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

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "runtime": {
            "package": "deface",
            "package_version": deface.__version__,
            "provider": provider,
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
    report_path = output_dir / "deface-run.json"
    _atomic_json(report_path, report)
    print(
        f"processed {len(jobs) - failures}/{len(jobs)} video(s); report: {report_path}"
    )
    return 1 if failures else 0


__all__ = [
    "NativeDefaceConfig",
    "main",
    "metadata_path_for",
    "run_native_deface_video",
]
