"""Detector-only face inventory for Hugging Face video datasets.

The scanner deliberately does not call the rendering or blurring pipeline. It
materializes one source video, decodes frames until the configured presence
criterion is met (or the video ends), and stores only detection metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from oopsie_backend.face_blur.benchmark import parse_input_size
from oopsie_backend.face_blur.cli import resolve_device
from oopsie_backend.face_blur.detectors import FaceDetector, create_detector
from oopsie_backend.face_blur.models import (
    MODEL_SPECS,
    default_model_dir,
    model_metadata,
    provision_model,
)
from oopsie_backend.utils.hf import parse_org_arg, parse_repo_arg

SCHEMA_VERSION = 1
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# These are camera suffixes used by Oopsie converters and existing submissions.
# The full repo-relative path is retained in all output, so this heuristic can
# be audited or replaced by an HDF5-derived mapping later without rerunning the
# detector.
CAMERA_SUFFIXES = tuple(
    sorted(
        {
            "camera_wrist_left",
            "camera_wrist_right",
            "camera_high",
            "cam_left_wrist",
            "cam_right_wrist",
            "cam_high",
            "left_shoulder_cam",
            "right_shoulder_cam",
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
            "left_wrist",
            "right_wrist",
            "exterior_cam",
            "image_primary",
            "overhead",
            "wrist",
            "main",
            "top",
            "left",
            "right",
        },
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True, slots=True)
class HfVideo:
    repo_id: str
    repo_revision: str
    source_path: str
    episode_key: str
    video_id: str


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_hash(data: dict) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_video_id(repo_id: str, source_path: str) -> str:
    digest = hashlib.sha256(f"{repo_id}\0{source_path}".encode()).hexdigest()[:16]
    return f"video_{digest}"


def video_shard(video: HfVideo, num_shards: int) -> int:
    """Assign a video to a stable, approximately balanced shard."""
    return int(video.video_id.removeprefix("video_"), 16) % num_shards


def infer_episode_key(source_path: str) -> str:
    """Infer a stable repo-relative episode key from an Oopsie video path.

    The mapping strips known camera suffixes but keeps parent directories and
    timestamps so repeated names such as ``episode_39`` remain distinct.
    """
    path = PurePosixPath(source_path)
    stem = path.stem
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    else:
        lower = stem.lower()
        for suffix in CAMERA_SUFFIXES:
            for separator in ("_", "-"):
                marker = f"{separator}{suffix}"
                if lower.endswith(marker):
                    stem = stem[: -len(marker)]
                    lower = stem.lower()
                    break
            else:
                continue
            break
    parent = path.parent
    return (parent / stem).as_posix() if str(parent) != "." else stem


def is_video_path(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


def _repo_revision(api: HfApi, repo_id: str, token: str | None, revision: str) -> str:
    info = api.dataset_info(
        repo_id=repo_id,
        revision=revision,
        token=token,
    )
    sha = getattr(info, "sha", None)
    if not sha:
        raise ValueError(f"Hugging Face returned no revision SHA for {repo_id}")
    return str(sha)


def resolve_dataset_repos(
    api: HfApi,
    *,
    repo: str | None,
    org: str | None,
    token: str | None,
) -> list[str]:
    if (repo is None) == (org is None):
        raise ValueError("provide exactly one of repo or org")
    if repo is not None:
        repo_id, repo_type = parse_repo_arg(repo, "dataset")
        if repo_type != "dataset":
            raise ValueError("face scanning supports Hugging Face dataset repos only")
        return [repo_id]

    assert org is not None
    org_id = parse_org_arg(org)
    repo_ids = []
    for info in api.list_datasets(author=org_id, token=token):
        repo_id = getattr(info, "id", None) or getattr(info, "dataset_id", None)
        if repo_id:
            repo_ids.append(str(repo_id))
    repo_ids = sorted(set(repo_ids))
    if not repo_ids:
        raise ValueError(
            f"no datasets visible under Hugging Face organization {org_id!r}; "
            "check that HF_TOKEN has organization read access"
        )
    return repo_ids


def discover_hf_videos(
    api: HfApi,
    repo_ids: Iterable[str],
    *,
    token: str | None,
    revision: str,
) -> list[HfVideo]:
    videos: list[HfVideo] = []
    for repo_id in repo_ids:
        resolved_revision = _repo_revision(api, repo_id, token, revision)
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            token=token,
        )
        for source_path in sorted(path for path in files if is_video_path(path)):
            videos.append(
                HfVideo(
                    repo_id=repo_id,
                    repo_revision=resolved_revision,
                    source_path=source_path,
                    episode_key=infer_episode_key(source_path),
                    video_id=stable_video_id(repo_id, source_path),
                )
            )
    return videos


def load_video_inventory(path: Path) -> list[HfVideo]:
    """Load and validate an inventory produced by a previous discovery pass."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    videos = []
    for row_number, row in enumerate(rows, start=2):
        try:
            video = HfVideo(
                repo_id=str(row["repo_id"]),
                repo_revision=str(row["repo_revision"]),
                source_path=str(row["source_path"]),
                episode_key=str(row["episode_key"]),
                video_id=str(row["video_id"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"inventory row {row_number} is missing column {exc.args[0]!r}"
            ) from exc
        expected_id = stable_video_id(video.repo_id, video.source_path)
        if video.video_id != expected_id:
            raise ValueError(
                f"inventory row {row_number} has video_id {video.video_id!r}; "
                f"expected {expected_id!r}"
            )
        if not video.repo_revision:
            raise ValueError(f"inventory row {row_number} has no repository revision")
        if not is_video_path(video.source_path):
            raise ValueError(
                f"inventory row {row_number} is not a supported video path: "
                f"{video.source_path!r}"
            )
        videos.append(video)
    if not videos:
        raise ValueError(f"inventory contains no videos: {path}")
    unique_ids = {video.video_id for video in videos}
    if len(unique_ids) != len(videos):
        raise ValueError(f"inventory contains duplicate video ids: {path}")
    return videos


def local_video_path(local_root: Path, video: HfVideo) -> Path:
    namespace, repo_name = video.repo_id.split("/", 1)
    return local_root.resolve() / namespace / repo_name / PurePosixPath(video.source_path)


@contextmanager
def materialize_video(
    video: HfVideo,
    *,
    token: str | None,
    download_mode: str,
    cache_dir: Path | None,
    temp_dir: Path | None,
    local_root: Path | None,
    downloader: Callable[..., str] = hf_hub_download,
) -> Iterator[tuple[Path, str]]:
    """Yield a local path and materialization mode for one immutable HF video."""
    if local_root is not None:
        path = local_video_path(local_root, video)
        if not path.is_file():
            raise FileNotFoundError(f"synced video not found: {path}")
        yield path, "local"
        return

    common = {
        "repo_id": video.repo_id,
        "filename": video.source_path,
        "repo_type": "dataset",
        "revision": video.repo_revision,
        "token": token,
    }
    if download_mode == "cache":
        path = Path(
            downloader(
                **common,
                cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
            )
        )
        yield path, "cache"
        return
    if download_mode != "temporary":
        raise ValueError(f"unsupported download mode: {download_mode}")

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="oopsie-face-scan-",
        dir=str(temp_dir.resolve()) if temp_dir is not None else None,
    ) as scratch:
        path = Path(downloader(**common, cache_dir=scratch))
        yield path, "temporary"


def scan_video(
    path: Path,
    detector: FaceDetector,
    *,
    score_threshold: float,
    min_positive_frames: int,
    min_consecutive_frames: int,
    stop_on_positive: bool,
    max_frames: int | None,
) -> dict:
    """Scan one video and return presence metadata without writing media."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    positive_frames = 0
    consecutive_frames = 0
    max_consecutive_frames = 0
    detection_count = 0
    max_score = 0.0
    first_positive_frame: int | None = None
    last_positive_frame: int | None = None
    stopped_on_positive = False
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            inference_started = time.perf_counter()
            detections = [
                detection
                for detection in detector.detect(frame)
                if float(detection.score) >= score_threshold
            ]
            inference_seconds += time.perf_counter() - inference_started
            if detections:
                positive_frames += 1
                consecutive_frames += 1
                max_consecutive_frames = max(
                    max_consecutive_frames, consecutive_frames
                )
                detection_count += len(detections)
                max_score = max(
                    max_score,
                    max(float(detection.score) for detection in detections),
                )
                if first_positive_frame is None:
                    first_positive_frame = frame_index
                last_positive_frame = frame_index
            else:
                consecutive_frames = 0
            frame_index += 1

            detected = (
                positive_frames >= min_positive_frames
                and max_consecutive_frames >= min_consecutive_frames
            )
            if detected and stop_on_positive:
                stopped_on_positive = True
                break
    finally:
        capture.release()

    elapsed_seconds = time.perf_counter() - started
    if frame_index == 0:
        raise RuntimeError(f"video decoded zero frames: {path}")
    has_face = (
        positive_frames >= min_positive_frames
        and max_consecutive_frames >= min_consecutive_frames
    )
    limited = max_frames is not None and frame_index >= max_frames
    scan_complete = not stopped_on_positive and not (
        limited and expected_frames > frame_index
    )
    return {
        "has_face": has_face,
        "scan_complete": scan_complete,
        "stopped_on_positive": stopped_on_positive,
        "width": width,
        "height": height,
        "fps": fps,
        "expected_frames": expected_frames,
        "frames_scanned": frame_index,
        "positive_frame_count": positive_frames,
        "max_consecutive_positive_frames": max_consecutive_frames,
        "detection_count": detection_count,
        "max_score": max_score,
        "first_positive_frame": first_positive_frame,
        "last_positive_frame": last_positive_frame,
        "elapsed_seconds": elapsed_seconds,
        "inference_seconds": inference_seconds,
        "pipeline_fps": frame_index / elapsed_seconds if elapsed_seconds else None,
        "inference_fps": (
            frame_index / inference_seconds if inference_seconds else None
        ),
    }


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            if fieldnames:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def can_resume(result: dict, video: HfVideo, config_hash: str) -> bool:
    source = result.get("source", {})
    return (
        result.get("status") == "ok"
        and result.get("config_hash") == config_hash
        and source.get("repo_id") == video.repo_id
        and source.get("repo_revision") == video.repo_revision
        and source.get("source_path") == video.source_path
    )


def result_to_video_row(result: dict) -> dict:
    source = result.get("source", {})
    scan = result.get("scan", {})
    status = result.get("status")
    return {
        "video_id": source.get("video_id"),
        "source_repo": source.get("repo_id"),
        "repo_revision": source.get("repo_revision"),
        "source_path": source.get("source_path"),
        "episode_key": source.get("episode_key"),
        "episode_mapping": "filename_heuristic",
        "status": status,
        "error": result.get("error"),
        "has_face": scan.get("has_face") if status == "ok" else None,
        "scan_complete": scan.get("scan_complete") if status == "ok" else None,
        "stopped_on_positive": (
            scan.get("stopped_on_positive") if status == "ok" else None
        ),
        "score_threshold": result.get("config", {}).get("score_threshold"),
        "max_score": scan.get("max_score"),
        "positive_frame_count": scan.get("positive_frame_count"),
        "max_consecutive_positive_frames": scan.get(
            "max_consecutive_positive_frames"
        ),
        "detection_count": scan.get("detection_count"),
        "first_positive_frame": scan.get("first_positive_frame"),
        "last_positive_frame": scan.get("last_positive_frame"),
        "frames_scanned": scan.get("frames_scanned"),
        "expected_frames": scan.get("expected_frames"),
        "width": scan.get("width"),
        "height": scan.get("height"),
        "fps": scan.get("fps"),
        "elapsed_seconds": scan.get("elapsed_seconds"),
        "inference_seconds": scan.get("inference_seconds"),
        "pipeline_fps": scan.get("pipeline_fps"),
        "inference_fps": scan.get("inference_fps"),
        "materialization": result.get("materialization"),
        "completed_at": result.get("completed_at"),
    }


def aggregate_episode_rows(video_rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in video_rows:
        groups[(str(row.get("source_repo")), str(row.get("episode_key")))].append(
            row
        )

    episode_rows = []
    for (repo_id, episode_key), rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("status") == "ok"]
        positive = [row for row in successful if row.get("has_face") is True]
        failed_count = len(rows) - len(successful)
        if positive:
            face_status = "detected"
        elif failed_count:
            face_status = "incomplete"
        else:
            face_status = "not_detected"
        max_scores = [
            float(row["max_score"])
            for row in successful
            if row.get("max_score") is not None
        ]
        episode_rows.append(
            {
                "source_repo": repo_id,
                "episode_key": episode_key,
                "episode_mapping": "filename_heuristic",
                "face_status": face_status,
                "has_face": True if positive else (None if failed_count else False),
                "video_count": len(rows),
                "successful_video_count": len(successful),
                "failed_video_count": failed_count,
                "positive_video_count": len(positive),
                "max_score": max(max_scores) if max_scores else None,
                "source_paths_json": json.dumps(
                    sorted(str(row.get("source_path")) for row in rows),
                    separators=(",", ":"),
                ),
                "positive_source_paths_json": json.dumps(
                    sorted(str(row.get("source_path")) for row in positive),
                    separators=(",", ":"),
                ),
            }
        )
    return episode_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="One Hugging Face dataset id or URL.")
    source.add_argument("--org", help="Scan every dataset visible under this HF org.")
    source.add_argument(
        "--inventory",
        type=Path,
        help="Reuse an immutable inventory.csv instead of relisting the Hub.",
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--download-mode",
        choices=("temporary", "cache"),
        default="temporary",
        help="Delete each downloaded video after scanning, or retain it in the HF cache.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Use an existing <root>/<namespace>/<repo>/ sync instead of downloading.",
    )
    parser.add_argument(
        "--detector", choices=list(MODEL_SPECS), default="deface-centerface"
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-size", type=parse_input_size, default=(640, 640))
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--min-consecutive-frames", type=int, default=1)
    parser.add_argument(
        "--no-stop-on-positive",
        action="store_true",
        help="Continue through the entire video after the presence criterion is met.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug only.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if not 0 <= args.nms_threshold <= 1:
        parser.error("--nms-threshold must be between 0 and 1")
    if args.min_positive_frames <= 0 or args.min_consecutive_frames <= 0:
        parser.error("minimum positive/consecutive frame counts must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.local_root is not None and args.download_mode != "temporary":
        parser.error("--download-mode is not used with --local-root")

    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=args.token)
    try:
        if args.inventory is not None:
            videos = load_video_inventory(args.inventory.resolve())
            repo_ids = sorted({video.repo_id for video in videos})
        else:
            repo_ids = resolve_dataset_repos(
                api,
                repo=args.repo,
                org=args.org,
                token=args.token,
            )
            videos = discover_hf_videos(
                api,
                repo_ids,
                token=args.token,
                revision=args.revision,
            )
    except Exception as exc:
        print(f"error: inventory: {type(exc).__name__}: {exc}")
        return 1
    videos = [
        video
        for video in videos
        if video_shard(video, args.num_shards) == args.shard_index
    ]
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print("error: no video files found")
        return 1

    inventory_rows = [asdict(video) for video in videos]
    _atomic_csv(output_dir / "inventory.csv", inventory_rows)

    device = resolve_device(args.device)
    try:
        model_path = provision_model(args.detector, args.model_dir)
        model_info = model_metadata(args.detector, model_path)
        detector = create_detector(
            args.detector,
            model_path,
            input_size=args.input_size,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            device=device,
        )
        detector.detect(
            np.zeros((args.input_size[1], args.input_size[0], 3), dtype=np.uint8)
        )
    except Exception as exc:
        print(f"error: detector: {type(exc).__name__}: {exc}")
        return 1

    config = {
        "schema_version": SCHEMA_VERSION,
        "detector": args.detector,
        "model_sha256": model_info["model_sha256"],
        "input_size": list(args.input_size),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "device": device,
        "min_positive_frames": args.min_positive_frames,
        "min_consecutive_frames": args.min_consecutive_frames,
        "stop_on_positive": not args.no_stop_on_positive,
        "max_frames": args.max_frames,
    }
    config_hash = canonical_hash(config)
    config["config_hash"] = config_hash
    _atomic_json(output_dir / "config.json", config)

    results = []
    failures = 0
    started = time.perf_counter()
    for video in tqdm(videos, desc="face presence", unit="video"):
        result_path = result_dir / f"{video.video_id}.json"
        if not args.no_resume and result_path.is_file():
            try:
                cached = _load_json(result_path)
            except (OSError, ValueError, json.JSONDecodeError):
                cached = {}
            if can_resume(cached, video, config_hash):
                results.append(cached)
                continue

        source = {
            "video_id": video.video_id,
            "repo_id": video.repo_id,
            "repo_revision": video.repo_revision,
            "source_path": video.source_path,
            "episode_key": video.episode_key,
        }
        try:
            with materialize_video(
                video,
                token=args.token,
                download_mode=args.download_mode,
                cache_dir=args.cache_dir,
                temp_dir=args.temp_dir,
                local_root=args.local_root,
            ) as (local_path, materialization):
                scan = scan_video(
                    local_path,
                    detector,
                    score_threshold=args.score_threshold,
                    min_positive_frames=args.min_positive_frames,
                    min_consecutive_frames=args.min_consecutive_frames,
                    stop_on_positive=not args.no_stop_on_positive,
                    max_frames=args.max_frames,
                )
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "config_hash": config_hash,
                "config": config,
                "source": source,
                "materialization": materialization,
                "scan": scan,
                "completed_at": utc_now(),
            }
        except Exception as exc:
            failures += 1
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "config_hash": config_hash,
                "config": config,
                "source": source,
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at": utc_now(),
            }
            tqdm.write(f"{video.repo_id}:{video.source_path}: {result['error']}")
        _atomic_json(result_path, result)
        results.append(result)

    video_rows = [result_to_video_row(result) for result in results]
    episode_rows = aggregate_episode_rows(video_rows)
    _atomic_csv(output_dir / "videos.csv", video_rows)
    _atomic_csv(output_dir / "episodes.csv", episode_rows)
    positive_videos = sum(row.get("has_face") is True for row in video_rows)
    positive_episodes = sum(row.get("has_face") is True for row in episode_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source": {
            "repo": args.repo,
            "org": args.org,
            "revision": args.revision,
            "inventory": str(args.inventory.resolve()) if args.inventory else None,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "config": config,
        "download": {
            "mode": "local" if args.local_root is not None else args.download_mode,
            "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
            "local_root": str(args.local_root.resolve()) if args.local_root else None,
        },
        "totals": {
            "repositories": len(repo_ids),
            "videos": len(video_rows),
            "episodes": len(episode_rows),
            "videos_with_faces": positive_videos,
            "episodes_with_faces": positive_episodes,
            "failed_videos": sum(row.get("status") != "ok" for row in video_rows),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "outputs": {
            "inventory": "inventory.csv",
            "videos": "videos.csv",
            "episodes": "episodes.csv",
            "results": "results/",
        },
    }
    _atomic_json(output_dir / "scan-report.json", report)
    print(
        f"scanned {len(video_rows) - failures}/{len(video_rows)} videos; "
        f"faces detected in {positive_videos} videos / {positive_episodes} episodes; "
        f"report: {output_dir / 'scan-report.json'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
