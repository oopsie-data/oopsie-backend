#!/usr/bin/env python3
"""Build a dataset-balanced public visualizer subset directly from Hugging Face.

Only eligible HDF5 metadata and the selected videos are downloaded. The full
source dataset is never materialized locally.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Callable, TypeVar

# Avoid one Xet read-token API request per small HDF5 file. Standard resolve
# downloads use a separate, more suitable Hub rate-limit bucket.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import h5py
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoFile

from oopsie_backend.scripts.export_public_visualizer_subset import (
    DATASET_WEIGHT_POWER,
    METADATA_FILENAME,
    MIN_FAILURE_PER_DATASET,
    MIN_SUCCESS_PER_DATASET,
    N_FAILURE,
    N_SUCCESS,
    POSTERS_DIRNAME,
    PUBLIC_VIDEO_KEYS,
    SAMPLING_STRATEGIES,
    SEED,
    VIDEOS_DIRNAME,
    allocate_dataset_quotas,
    allocate_outcomes_within_total_quotas,
    build_public_filters,
    has_face_candidate,
    load_face_index,
    outcome,
    print_dataset_quotas,
    public_info_html,
    rebalance_total_quotas_for_outcome_target,
    sample_by_dataset_quota,
)
from oopsie_backend.scripts.generate_visualizer_assets import (
    filter_id,
    make_poster,
    read_attr,
    read_failure_annotation,
    success_label,
    transcode_video,
)
from oopsie_backend.utils.episode_stats import (
    HF_ORG,
    PUBLIC_EXCLUDED_DATASETS,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "zhouzypaul/oopsie-data"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "assets/visualizer-public-hf"
DEFAULT_CACHE_DIR = BACKEND_ROOT / "assets/hf-cache"
DEFAULT_FACE_EPISODES_CSV = BACKEND_ROOT / "reports/face_candidate_episodes.csv"
DEFAULT_FACE_VIDEOS_CSV = BACKEND_ROOT / "reports/face_candidate_videos.csv"
HDF5_SUFFIXES = {".h5", ".hdf5"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
T = TypeVar("T")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--face-episodes-csv", type=Path, default=DEFAULT_FACE_EPISODES_CSV)
    parser.add_argument("--face-videos-csv", type=Path, default=DEFAULT_FACE_VIDEOS_CSV)
    parser.add_argument("--n-failure", type=int, default=N_FAILURE)
    parser.add_argument("--n-success", type=int, default=N_SUCCESS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--min-failure-per-dataset",
        type=int,
        default=MIN_FAILURE_PER_DATASET,
    )
    parser.add_argument(
        "--min-success-per-dataset",
        type=int,
        default=MIN_SUCCESS_PER_DATASET,
    )
    parser.add_argument(
        "--dataset-weight-power",
        type=float,
        default=DATASET_WEIGHT_POWER,
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="outcome-weighted",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel selected-video downloads and ffmpeg jobs.",
    )
    parser.add_argument(
        "--metadata-workers",
        type=int,
        default=1,
        help=(
            "Parallel HDF5 metadata downloads. Keep at 1 for private repos "
            "with the default Hugging Face API rate limit."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF read token; defaults to $HF_TOKEN or the cached HF login.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed videos/posters in an existing output directory.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.metadata_workers < 1:
        raise SystemExit("--metadata-workers must be at least 1")
    if args.output_root.exists() and not args.resume:
        raise SystemExit(
            f"Output already exists: {args.output_root}. "
            "Move/remove it, or pass --resume to reuse completed files."
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=args.token)
    resolved_revision = resolve_revision(
        api=api,
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
    )
    print(
        f"Reading {args.repo_id}@{resolved_revision} with cached/selective downloads",
        flush=True,
    )

    repo_files = load_or_fetch_repo_manifest(
        api,
        repo_id=args.repo_id,
        revision=resolved_revision,
        token=args.token,
        cache_dir=args.cache_dir,
    )
    eligible_h5_paths = sorted(
        path
        for path in repo_files
        if PurePosixPath(path).suffix.lower() in HDF5_SUFFIXES
        and not is_excluded_dataset_path(path)
    )
    if not eligible_h5_paths:
        raise SystemExit("No eligible HDF5 episodes found")

    eligible_datasets = sorted(
        {PurePosixPath(path).parts[0] for path in eligible_h5_paths}
    )
    print(
        f"Found {len(eligible_h5_paths)} eligible HDF5 files "
        f"across {len(eligible_datasets)} datasets",
        flush=True,
    )
    metadata_root = download_metadata_snapshot(
        repo_id=args.repo_id,
        revision=resolved_revision,
        datasets=eligible_datasets,
        cache_dir=args.cache_dir,
        token=args.token,
        workers=args.metadata_workers,
    )

    face_index = load_face_index(args.face_episodes_csv, args.face_videos_csv)
    candidates = read_episode_candidates(
        metadata_root=metadata_root,
        h5_paths=eligible_h5_paths,
        repo_files=repo_files,
        face_index=face_index,
    )
    candidates, removed_cmu_tactile_only = apply_cmu_camera_policy(candidates)
    print(
        "CMU camera policy: removed tactile streams; excluded "
        f"{removed_cmu_tactile_only} tactile-only episodes",
        flush=True,
    )
    failures = [candidate for candidate in candidates if outcome(candidate) == "failure"]
    successes = [candidate for candidate in candidates if outcome(candidate) == "success"]
    print(
        f"Eligible face-safe episodes: {len(candidates)} total, "
        f"{len(failures)} failures, {len(successes)} successes",
        flush=True,
    )

    try:
        if args.sampling_strategy == "equal-total":
            total_quotas = allocate_dataset_quotas(
                failures + successes,
                target=args.n_failure + args.n_success,
                minimum=0,
                weight_power=0,
            )
            total_quotas = rebalance_total_quotas_for_outcome_target(
                failures,
                successes,
                total_quotas,
                target_success=args.n_success,
            )
            failure_quotas, success_quotas = allocate_outcomes_within_total_quotas(
                failures,
                successes,
                total_quotas,
                target_success=args.n_success,
            )
        else:
            failure_quotas = allocate_dataset_quotas(
                failures,
                target=args.n_failure,
                minimum=args.min_failure_per_dataset,
                weight_power=args.dataset_weight_power,
            )
            success_quotas = allocate_dataset_quotas(
                successes,
                target=args.n_success,
                minimum=args.min_success_per_dataset,
                weight_power=args.dataset_weight_power,
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print_dataset_quotas("failure", failure_quotas)
    print_dataset_quotas("success", success_quotas)
    rng = random.Random(args.seed)
    sampled = sample_by_dataset_quota(failures, failure_quotas, rng)
    sampled.extend(sample_by_dataset_quota(successes, success_quotas, rng))
    rng.shuffle(sampled)
    camera_replacements: Counter[tuple[str, str, str]] = Counter()
    for candidate in sampled:
        camera_name, remote_video_path, replaced_camera = select_camera_path(
            candidate,
            rng,
        )
        if replaced_camera is not None:
            camera_replacements[
                (candidate["dataset_name"], replaced_camera, camera_name)
            ] += 1
        candidate["selected_camera_name"] = camera_name
        candidate["selected_video_path"] = remote_video_path
    for (dataset, old_camera, new_camera), count in sorted(
        camera_replacements.items()
    ):
        print(
            f"Camera preference: {dataset} {old_camera} -> {new_camera}: "
            f"{count} episodes",
            flush=True,
        )

    videos_out = args.output_root / VIDEOS_DIRNAME
    posters_out = args.output_root / POSTERS_DIRNAME
    videos_out.mkdir(parents=True, exist_ok=args.resume)
    posters_out.mkdir(parents=True, exist_ok=args.resume)

    def process_indexed(item: tuple[int, dict]) -> dict[str, object]:
        index, candidate = item
        return materialize_public_video(
            index=index,
            candidate=candidate,
            repo_id=args.repo_id,
            revision=resolved_revision,
            cache_dir=args.cache_dir,
            token=args.token,
            output_root=args.output_root,
        )

    indexed = list(enumerate(sampled, start=1))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        public_videos = list(executor.map(process_indexed, indexed))

    public_metadata = {
        "videos": public_videos,
        "filters": build_public_filters(public_videos),
        "source": {
            "revision": resolved_revision,
            "seed": args.seed,
            "n_failure": args.n_failure,
            "n_success": args.n_success,
            "n_total": len(public_videos),
            "face_filter": True,
            "dataset_sampling": {
                "strategy": args.sampling_strategy,
                "weight_power": args.dataset_weight_power,
                "min_failure_per_dataset": args.min_failure_per_dataset,
                "min_success_per_dataset": args.min_success_per_dataset,
                "represented_datasets": len(
                    {
                        name
                        for name, quota in failure_quotas.items()
                        if quota > 0
                    }
                    | {
                        name
                        for name, quota in success_quotas.items()
                        if quota > 0
                    }
                ),
            },
        },
    }
    metadata_path = args.output_root / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(public_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(public_videos)} clips to {args.output_root}", flush=True)


def list_repo_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | None,
) -> set[str]:
    files: set[str] = set()
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=False,
        token=token,
    ):
        if isinstance(entry, RepoFile):
            files.add(entry.path)
    return files


def is_excluded_dataset_path(remote_path: str) -> bool:
    parts = PurePosixPath(remote_path).parts
    if not parts:
        return True
    return f"{HF_ORG}/{parts[0]}" in PUBLIC_EXCLUDED_DATASETS


def download_metadata_snapshot(
    repo_id: str,
    revision: str,
    datasets: list[str],
    cache_dir: Path,
    token: str | None,
    workers: int,
) -> Path:
    patterns: list[str] = []
    for dataset in datasets:
        patterns.extend(
            [
                f"{dataset}/*.h5",
                f"{dataset}/**/*.h5",
                f"{dataset}/*.hdf5",
                f"{dataset}/**/*.hdf5",
            ]
        )
    snapshot_path = with_rate_limit_retry(
        "metadata snapshot",
        lambda: snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=patterns,
            max_workers=workers,
        ),
    )
    return Path(snapshot_path)


def read_episode_candidates(
    metadata_root: Path,
    h5_paths: list[str],
    repo_files: set[str],
    face_index: dict[str, dict[str, set[str]]],
) -> list[dict]:
    candidates: list[dict] = []
    skipped_missing_metadata = 0
    skipped_no_video = 0
    for index, remote_h5_path in enumerate(h5_paths, start=1):
        local_h5_path = metadata_root / remote_h5_path
        if not local_h5_path.is_file():
            skipped_missing_metadata += 1
            continue

        dataset = PurePosixPath(remote_h5_path).parts[0]
        dataset_repo = f"{HF_ORG}/{dataset}"
        try:
            with h5py.File(local_h5_path, "r") as h5:
                episode_id = read_attr(h5, "episode_id", local_h5_path.stem)
                language_instruction = read_attr(
                    h5,
                    "language_instruction",
                    local_h5_path.stem,
                )
                annotation = read_failure_annotation(h5)
                camera_paths = read_remote_camera_paths(
                    h5=h5,
                    remote_h5_path=remote_h5_path,
                    repo_files=repo_files,
                    dataset_repo=dataset_repo,
                    episode_id=episode_id,
                    face_index=face_index,
                )
        except (OSError, KeyError, ValueError) as error:
            print(f"Skipping unreadable {remote_h5_path}: {error}", flush=True)
            continue

        if not camera_paths:
            skipped_no_video += 1
            continue
        candidates.append(
            {
                "dataset_name": dataset,
                "dataset_repo": dataset_repo,
                "h5_path": remote_h5_path,
                "episode_id": episode_id,
                "language_instruction": language_instruction,
                "success": annotation["success"],
                "annotation": annotation,
                "camera_paths": camera_paths,
            }
        )
        if index % 500 == 0:
            print(f"Read metadata for {index}/{len(h5_paths)} episodes", flush=True)

    print(
        f"Metadata skips: {skipped_missing_metadata} missing, "
        f"{skipped_no_video} without a face-safe video",
        flush=True,
    )
    return deduplicate_candidates(candidates)


def read_remote_camera_paths(
    h5: h5py.File,
    remote_h5_path: str,
    repo_files: set[str],
    dataset_repo: str,
    episode_id: str,
    face_index: dict[str, dict[str, set[str]]],
) -> list[tuple[str, str]]:
    if "observations/video_paths" not in h5:
        return []

    camera_paths: list[tuple[str, str]] = []
    video_paths = h5["observations/video_paths"]
    for camera_name in sorted(video_paths.keys()):
        raw_path = video_paths[camera_name][()]
        if isinstance(raw_path, bytes):
            raw_path = raw_path.decode("utf-8", errors="replace")
        remote_video_path = resolve_remote_video_path(
            remote_h5_path,
            str(raw_path),
            repo_files,
        )
        if remote_video_path is None:
            continue
        probe = {
            "dataset_repo": dataset_repo,
            "episode_id": episode_id,
            "src": remote_video_path,
        }
        if has_face_candidate(probe, face_index):
            continue
        camera_paths.append((camera_name, remote_video_path))
    return camera_paths


def resolve_remote_video_path(
    remote_h5_path: str,
    raw_video_path: str,
    repo_files: set[str],
) -> str | None:
    h5_parent = posixpath.dirname(remote_h5_path)
    raw = raw_video_path.strip().replace("\\", "/")
    candidates: list[str] = []
    if raw and not raw.startswith("/"):
        candidates.append(posixpath.normpath(posixpath.join(h5_parent, raw)))
    if raw:
        candidates.append(posixpath.join(h5_parent, posixpath.basename(raw)))

    dataset = PurePosixPath(remote_h5_path).parts[0]
    raw_parts = PurePosixPath(raw).parts
    if dataset in raw_parts:
        dataset_index = raw_parts.index(dataset)
        candidates.append(PurePosixPath(*raw_parts[dataset_index:]).as_posix())

    for candidate in dict.fromkeys(candidates):
        if (
            candidate in repo_files
            and PurePosixPath(candidate).suffix.lower() in VIDEO_SUFFIXES
        ):
            return candidate
    return None


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        key = (candidate["dataset_name"], candidate["episode_id"])
        if key not in unique:
            unique[key] = candidate
    return list(unique.values())


def select_camera_path(
    candidate: dict,
    rng: random.Random,
) -> tuple[str, str, str | None]:
    """Select one camera, preferring CMU front and rejecting tactile output."""
    camera_name, remote_video_path = rng.choice(candidate["camera_paths"])
    if candidate["dataset_name"] != "CMU_Intent":
        return camera_name, remote_video_path, None

    camera_paths = dict(candidate["camera_paths"])
    front_path = camera_paths.get("front")
    if front_path is not None:
        replaced_camera = camera_name if camera_name != "front" else None
        return "front", front_path, replaced_camera
    if camera_name != "tactile":
        return camera_name, remote_video_path, None

    for preferred_name in ("front", "wrist", "zed", "exterior", "left", "right"):
        preferred_path = camera_paths.get(preferred_name)
        if preferred_path is not None:
            return preferred_name, preferred_path, camera_name

    raise ValueError(
        f"CMU_Intent episode {candidate['episode_id']!r} has only a tactile "
        "face-safe view"
    )


def apply_cmu_camera_policy(candidates: list[dict]) -> tuple[list[dict], int]:
    """Remove CMU tactile views and episodes with no remaining visual stream."""
    kept: list[dict] = []
    removed_tactile_only = 0
    for candidate in candidates:
        if candidate["dataset_name"] != "CMU_Intent":
            kept.append(candidate)
            continue
        visual_paths = [
            camera_path
            for camera_path in candidate["camera_paths"]
            if camera_path[0] != "tactile"
        ]
        if not visual_paths:
            removed_tactile_only += 1
            continue
        candidate["camera_paths"] = visual_paths
        kept.append(candidate)
    return kept, removed_tactile_only


def materialize_public_video(
    index: int,
    candidate: dict,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    token: str | None,
    output_root: Path,
) -> dict[str, object]:
    opaque_id = f"ep_{index:04d}"
    source_path = Path(
        with_rate_limit_retry(
            f"video {candidate['selected_video_path']}",
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=candidate["selected_video_path"],
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            ),
        )
    )
    output_video = output_root / VIDEOS_DIRNAME / f"{opaque_id}.mp4"
    output_poster = output_root / POSTERS_DIRNAME / f"{opaque_id}.jpg"
    transcode_video(source_path, output_video, shutil.which("ffmpeg") or "ffmpeg")
    make_poster(output_video, output_poster, shutil.which("ffmpeg") or "ffmpeg")

    camera_name = candidate["selected_camera_name"]
    annotation = candidate["annotation"]
    task_label = success_label(candidate["success"])
    scrubbed = {
        "src": f"{VIDEOS_DIRNAME}/{opaque_id}.mp4",
        "poster": f"{POSTERS_DIRNAME}/{opaque_id}.jpg",
        "info": public_info_html(
            candidate["language_instruction"],
            camera_name,
            task_label,
        ),
        "camera_id": filter_id(camera_name, "camera"),
        "camera_label": camera_name,
        "task_id": filter_id(task_label, "task"),
        "task_label": task_label,
        "annotation_id": filter_id(annotation["annotation_label"], "annotation"),
        "annotation_label": annotation["annotation_label"],
        "failure_category_ids": [
            filter_id(category, "failure_category")
            for category in annotation["failure_categories"]
        ],
        "failure_category_labels": annotation["failure_categories"],
        "severity_id": filter_id(annotation["severity_label"], "severity"),
        "severity_label": annotation["severity_label"],
        "failure_description": annotation["failure_description"],
        "additional_notes": annotation["additional_notes"],
        "episode_id": opaque_id,
        "success": candidate["success"],
    }
    print(
        f"[{index}] {candidate['selected_video_path']} -> {output_video.name}",
        flush=True,
    )
    return {key: scrubbed[key] for key in PUBLIC_VIDEO_KEYS}


def resolve_revision(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | None,
) -> str:
    if COMMIT_PATTERN.fullmatch(revision):
        return revision
    repo_info = with_rate_limit_retry(
        "repository revision",
        lambda: api.dataset_info(
            repo_id=repo_id,
            revision=revision,
            token=token,
        ),
    )
    return repo_info.sha


def load_or_fetch_repo_manifest(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | None,
    cache_dir: Path,
) -> set[str]:
    manifest_dir = cache_dir / "visualizer-manifests"
    manifest_path = manifest_dir / f"{revision}.json"
    if manifest_path.is_file():
        return set(json.loads(manifest_path.read_text()))

    files = with_rate_limit_retry(
        "repository file manifest",
        lambda: list_repo_files(
            api,
            repo_id=repo_id,
            revision=revision,
            token=token,
        ),
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(sorted(files)) + "\n")
    temp_path.replace(manifest_path)
    return files


def with_rate_limit_retry(
    label: str,
    operation: Callable[[], T],
    max_attempts: int = 12,
) -> T:
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except HfHubHTTPError as error:
            response = error.response
            if response is None or response.status_code != 429 or attempt == max_attempts:
                raise
            delay = rate_limit_delay_seconds(error)
            print(
                f"Hugging Face rate limit while fetching {label}; "
                f"retrying in {delay}s (attempt {attempt}/{max_attempts})",
                flush=True,
            )
            remaining = delay
            while remaining > 0:
                chunk = min(remaining, 60)
                time.sleep(chunk)
                remaining -= chunk
    raise RuntimeError(f"Exhausted retries for {label}")


def rate_limit_delay_seconds(error: HfHubHTTPError) -> int:
    response = error.response
    if response is not None:
        rate_limit = response.headers.get("ratelimit", "")
        match = re.search(r"(?:^|;)t=(\d+)", rate_limit)
        if match:
            return max(5, int(match.group(1)) + 5)
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.isdigit():
            return max(5, int(retry_after) + 5)
    return 305


if __name__ == "__main__":
    main()
