#!/usr/bin/env python3
"""Export a privacy-scrubbed public visualizer subset for Hugging Face.

Samples 400 failures + 100 successes from non-bulk datasets only (one camera
per episode), excludes face-candidate episodes/videos, anonymizes ids/paths/labels,
and copies only those media files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from oopsie_backend.utils.episode_stats import PUBLIC_EXCLUDED_DATASETS

DEFAULT_INPUT_ROOT = Path(
    "/home/arpit/test_projects/oopsiedata/oopsie-docs/docs/assets/visualizer"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/arpit/test_projects/oopsiedata/oopsie-docs/docs/assets/visualizer-public"
)
DEFAULT_FACE_EPISODES_CSV = Path(
    "/home/arpit/test_projects/oopsiedata/oopsie-backend/reports/face_candidate_episodes.csv"
)
DEFAULT_FACE_VIDEOS_CSV = Path(
    "/home/arpit/test_projects/oopsiedata/oopsie-backend/reports/face_candidate_videos.csv"
)

N_FAILURE = 400
N_SUCCESS = 100
SEED = 42
MIN_FAILURE_PER_DATASET = 10
MIN_SUCCESS_PER_DATASET = 3
DATASET_WEIGHT_POWER = 0.5
SAMPLING_STRATEGIES = ("outcome-weighted", "equal-total")

METADATA_FILENAME = "metadata.json"
VIDEOS_DIRNAME = "videos"
POSTERS_DIRNAME = "posters"

# Fields allowed in the public JSON (no repo / university identifiers).
PUBLIC_VIDEO_KEYS = (
    "src",
    "poster",
    "info",
    "camera_id",
    "camera_label",
    "task_id",
    "task_label",
    "annotation_id",
    "annotation_label",
    "failure_category_ids",
    "failure_category_labels",
    "severity_id",
    "severity_label",
    "failure_description",
    "additional_notes",
    "episode_id",
    "success",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--face-episodes-csv", type=Path, default=DEFAULT_FACE_EPISODES_CSV)
    parser.add_argument("--face-videos-csv", type=Path, default=DEFAULT_FACE_VIDEOS_CSV)
    parser.add_argument("--n-failure", type=int, default=N_FAILURE)
    parser.add_argument("--n-success", type=int, default=N_SUCCESS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--min-failure-per-dataset",
        type=int,
        default=MIN_FAILURE_PER_DATASET,
        help="Minimum failure samples per dataset when enough are available.",
    )
    parser.add_argument(
        "--min-success-per-dataset",
        type=int,
        default=MIN_SUCCESS_PER_DATASET,
        help="Minimum success samples per dataset when enough are available.",
    )
    parser.add_argument(
        "--dataset-weight-power",
        type=float,
        default=DATASET_WEIGHT_POWER,
        help=(
            "Dataset-size exponent for distributing samples above the minimum: "
            "0 is equal, 0.5 is square-root balanced, and 1 is proportional."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="outcome-weighted",
        help=(
            "outcome-weighted allocates failures/successes separately; equal-total "
            "first gives datasets capacity-bounded equal total quotas."
        ),
    )
    args = parser.parse_args()

    metadata_path = args.input_root / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SystemExit(
            f"Missing {metadata_path}. Re-run generate_visualizer_assets.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    videos = metadata.get("videos") or []
    excluded = set(metadata.get("excluded_from_public_datasets") or []) | set(
        PUBLIC_EXCLUDED_DATASETS
    )

    face_index = load_face_index(args.face_episodes_csv, args.face_videos_csv)

    public_clips = [v for v in videos if not is_bulk(v, excluded)]
    before_face = len(public_clips)
    public_clips = [v for v in public_clips if not has_face_candidate(v, face_index)]
    removed_faces = before_face - len(public_clips)
    print(
        f"Non-bulk clips: {before_face}; removed face candidates: {removed_faces}; "
        f"remaining: {len(public_clips)}",
        flush=True,
    )

    episode_clips = pick_one_camera_per_episode(public_clips, args.seed)

    failures = [v for v in episode_clips if outcome(v) == "failure"]
    successes = [v for v in episode_clips if outcome(v) == "success"]

    rng = random.Random(args.seed)
    if len(failures) < args.n_failure:
        raise SystemExit(
            f"Need {args.n_failure} failure episodes, found {len(failures)}"
        )
    if len(successes) < args.n_success:
        raise SystemExit(
            f"Need {args.n_success} success episodes, found {len(successes)}"
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
    sampled = sample_by_dataset_quota(failures, failure_quotas, rng)
    sampled.extend(sample_by_dataset_quota(successes, success_quotas, rng))
    rng.shuffle(sampled)

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    videos_out = args.output_root / VIDEOS_DIRNAME
    posters_out = args.output_root / POSTERS_DIRNAME
    videos_out.mkdir(parents=True)
    posters_out.mkdir(parents=True)

    public_videos: list[dict[str, object]] = []
    for index, video in enumerate(sampled, start=1):
        opaque_id = f"ep_{index:04d}"
        src_name = f"{opaque_id}.mp4"
        poster_name = f"{opaque_id}.jpg"

        src_rel = str(video.get("src") or "")
        poster_rel = str(video.get("poster") or "")
        src_path = args.input_root / src_rel
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source video: {src_path}")

        shutil.copy2(src_path, videos_out / src_name)
        poster_out_rel = ""
        if poster_rel:
            poster_path = args.input_root / poster_rel
            if poster_path.is_file():
                shutil.copy2(poster_path, posters_out / poster_name)
                poster_out_rel = f"{POSTERS_DIRNAME}/{poster_name}"

        instruction = instruction_from_info(str(video.get("info") or ""))
        camera = str(video.get("camera_label") or "camera")
        task = str(video.get("task_label") or outcome_label(video))
        scrubbed = {
            "src": f"{VIDEOS_DIRNAME}/{src_name}",
            "poster": poster_out_rel,
            "info": public_info_html(instruction, camera, task),
            "camera_id": video.get("camera_id"),
            "camera_label": video.get("camera_label"),
            "task_id": video.get("task_id"),
            "task_label": video.get("task_label"),
            "annotation_id": video.get("annotation_id"),
            "annotation_label": video.get("annotation_label"),
            "failure_category_ids": video.get("failure_category_ids") or [],
            "failure_category_labels": video.get("failure_category_labels") or [],
            "severity_id": video.get("severity_id"),
            "severity_label": video.get("severity_label"),
            "failure_description": video.get("failure_description") or "",
            "additional_notes": video.get("additional_notes") or "",
            "episode_id": opaque_id,
            "success": video.get("success"),
        }
        public_videos.append({key: scrubbed[key] for key in PUBLIC_VIDEO_KEYS})

    public_metadata = {
        "videos": public_videos,
        "filters": build_public_filters(public_videos),
        "source": {
            "seed": args.seed,
            "n_failure": args.n_failure,
            "n_success": args.n_success,
            "n_total": len(public_videos),
            "face_filter": True,
            "face_candidates_removed": removed_faces,
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
    out_meta = args.output_root / METADATA_FILENAME
    out_meta.write_text(json.dumps(public_metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(public_videos)} scrubbed clips "
        f"({args.n_failure} failure + {args.n_success} success) to {args.output_root}"
    )


def load_face_index(
    episodes_csv: Path, videos_csv: Path
) -> dict[str, dict[str, set[str]]]:
    """Return per-repo blocked episode tokens and video basenames/stems."""
    index: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"episode_tokens": set(), "video_stems": set(), "video_basenames": set()}
    )

    if episodes_csv.is_file():
        with episodes_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                repo = (row.get("source_repo") or "").strip()
                key = (row.get("episode_key") or "").strip()
                if not repo or not key:
                    continue
                index[repo]["episode_tokens"].add(key)
                index[repo]["episode_tokens"].add(Path(key).name)
    else:
        print(f"Warning: missing face episodes CSV: {episodes_csv}", flush=True)

    if videos_csv.is_file():
        with videos_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                repo = (row.get("source_repo") or "").strip()
                source_path = (row.get("source_path") or "").strip()
                episode_key = (row.get("episode_key") or "").strip()
                if not repo:
                    continue
                if episode_key:
                    index[repo]["episode_tokens"].add(episode_key)
                    index[repo]["episode_tokens"].add(Path(episode_key).name)
                if source_path:
                    path = Path(source_path)
                    index[repo]["video_basenames"].add(path.name)
                    index[repo]["video_stems"].add(path.stem)
                    index[repo]["episode_tokens"].add(path.stem)
    else:
        print(f"Warning: missing face videos CSV: {videos_csv}", flush=True)

    return index


def has_face_candidate(video: dict, face_index: dict[str, dict[str, set[str]]]) -> bool:
    repo = str(video.get("dataset_repo") or "")
    if not repo or repo not in face_index:
        return False

    blocked = face_index[repo]
    episode_id = str(video.get("episode_id") or "")
    src = str(video.get("src") or "")
    src_name = Path(src).name
    src_stem = Path(src).stem  # e.g. 005a30b4c04a__right_shoulder_cam

    if episode_id and episode_id in blocked["episode_tokens"]:
        return True

    # Visualizer stems are "{episode_or_file_stem}__{camera}".
    stem_prefix = src_stem.split("__", 1)[0] if src_stem else ""
    if stem_prefix and stem_prefix in blocked["episode_tokens"]:
        return True
    if src_stem and src_stem in blocked["episode_tokens"]:
        return True

    if src_name and src_name in blocked["video_basenames"]:
        return True
    if src_stem and src_stem in blocked["video_stems"]:
        return True
    if stem_prefix and stem_prefix in blocked["video_stems"]:
        return True

    # Path-style episode keys like videos/episode_000000 vs stem episode_000000__cam.
    for token in blocked["episode_tokens"]:
        token_name = Path(token).name
        if not token_name:
            continue
        if episode_id and (episode_id == token_name or token_name in episode_id):
            return True
        if stem_prefix and (
            stem_prefix == token_name
            or stem_prefix.startswith(token_name + "_")
            or token_name.startswith(stem_prefix)
        ):
            return True

    return False


def is_bulk(video: dict, excluded: set[str]) -> bool:
    if video.get("bulk_import"):
        return True
    repo = str(video.get("dataset_repo") or "")
    return bool(repo) and repo in excluded


def outcome(video: dict) -> str | None:
    success = video.get("success")
    if success is None:
        return None
    return "success" if float(success) >= 0.5 else "failure"


def outcome_label(video: dict) -> str:
    value = outcome(video)
    if value == "success":
        return "Success"
    if value == "failure":
        return "Failure"
    return "Unknown"


def dataset_name(video: dict) -> str:
    """Return the private sampling stratum without adding it to public metadata."""
    explicit = str(video.get("dataset_name") or "").strip()
    if explicit:
        return explicit

    for path_key in ("hf_path", "h5_path"):
        remote_path = str(video.get(path_key) or "").strip()
        if remote_path:
            parts = PurePosixPath(remote_path).parts
            if parts:
                return parts[0]

    dataset_repo = str(video.get("dataset_repo") or "").strip().rstrip("/")
    if dataset_repo:
        return dataset_repo.rsplit("/", 1)[-1]

    # Backward-compatible fallback for generated paths such as
    # videos/Armnet/episode_000000__front.mp4.
    src_parts = PurePosixPath(str(video.get("src") or "")).parts
    if len(src_parts) >= 3 and src_parts[0] in {VIDEOS_DIRNAME, POSTERS_DIRNAME}:
        return src_parts[1]

    raise ValueError(
        "Cannot determine dataset for episode "
        f"{video.get('episode_id')!r}; expected dataset_name, hf_path, h5_path, "
        "dataset_repo, or a dataset folder in src."
    )


def allocate_dataset_quotas(
    episodes: list[dict],
    target: int,
    minimum: int,
    weight_power: float = DATASET_WEIGHT_POWER,
) -> dict[str, int]:
    """Allocate an exact, capacity-bounded target across dataset strata."""
    if target < 0:
        raise ValueError("Sample target must be non-negative")
    if minimum < 0:
        raise ValueError("Per-dataset minimum must be non-negative")
    if weight_power < 0:
        raise ValueError("Dataset weight power must be non-negative")

    capacities = Counter(dataset_name(episode) for episode in episodes)
    available = sum(capacities.values())
    if available < target:
        raise ValueError(f"Need {target} episodes, found {available}")
    if target == 0:
        return {}

    quotas = {
        name: min(minimum, capacity)
        for name, capacity in sorted(capacities.items())
    }
    minimum_total = sum(quotas.values())
    if minimum_total > target:
        raise ValueError(
            f"Per-dataset minimum requires {minimum_total} samples, "
            f"but the target is {target}; lower the minimum"
        )

    # Weighted fair allocation of the remaining slots. Tracking additions above
    # the floor prevents the floor itself from distorting the requested weights.
    additions = {name: 0 for name in quotas}
    weights = {
        name: capacities[name] ** weight_power
        for name in quotas
    }
    remaining = target - minimum_total
    while remaining:
        eligible = [
            name
            for name in quotas
            if quotas[name] < capacities[name]
        ]
        if not eligible:
            raise ValueError(
                f"Could not allocate {target} samples within dataset capacities"
            )

        # Highest weight per next added slot wins. Name is an explicit,
        # deterministic tie-breaker independent of input ordering.
        selected = min(
            eligible,
            key=lambda name: (
                -(weights[name] / (additions[name] + 1)),
                name,
            ),
        )
        quotas[selected] += 1
        additions[selected] += 1
        remaining -= 1

    return quotas


def sample_by_dataset_quota(
    episodes: list[dict],
    quotas: dict[str, int],
    rng: random.Random,
) -> list[dict]:
    """Sample the requested number of episodes independently per dataset."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        grouped[dataset_name(episode)].append(episode)

    sampled: list[dict] = []
    for name in sorted(quotas):
        quota = quotas[name]
        if quota > len(grouped[name]):
            raise ValueError(
                f"Dataset {name!r} has {len(grouped[name])} episodes, "
                f"below its quota of {quota}"
            )
        sampled.extend(rng.sample(grouped[name], quota))
    return sampled


def allocate_outcomes_within_total_quotas(
    failures: list[dict],
    successes: list[dict],
    total_quotas: dict[str, int],
    target_success: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Fit an exact success target inside fixed per-dataset total quotas."""
    failure_capacities = Counter(dataset_name(episode) for episode in failures)
    success_capacities = Counter(dataset_name(episode) for episode in successes)

    lower_success: dict[str, int] = {}
    upper_success: dict[str, int] = {}
    for name, total_quota in total_quotas.items():
        total_capacity = failure_capacities[name] + success_capacities[name]
        if total_quota > total_capacity:
            raise ValueError(
                f"Dataset {name!r} has capacity {total_capacity}, "
                f"below its total quota of {total_quota}"
            )
        lower_success[name] = max(
            0,
            total_quota - failure_capacities[name],
        )
        upper_success[name] = min(
            total_quota,
            success_capacities[name],
        )

    minimum_success = sum(lower_success.values())
    maximum_success = sum(upper_success.values())
    if not minimum_success <= target_success <= maximum_success:
        raise ValueError(
            f"Cannot fit {target_success} successes inside total dataset quotas; "
            f"feasible range is {minimum_success}..{maximum_success}"
        )

    success_quotas = dict(lower_success)
    remaining = target_success - minimum_success
    while remaining:
        eligible = [
            name
            for name in sorted(total_quotas)
            if success_quotas[name] < upper_success[name]
        ]
        # Give the next success to the dataset with the lowest resulting
        # success fraction, keeping outcome mix as even as capacities allow.
        selected = min(
            eligible,
            key=lambda name: (
                (success_quotas[name] + 1) / total_quotas[name],
                name,
            ),
        )
        success_quotas[selected] += 1
        remaining -= 1

    failure_quotas = {
        name: total_quotas[name] - success_quotas[name]
        for name in total_quotas
    }
    return failure_quotas, success_quotas


def rebalance_total_quotas_for_outcome_target(
    failures: list[dict],
    successes: list[dict],
    total_quotas: dict[str, int],
    target_success: int,
) -> dict[str, int]:
    """Minimally shift equal total quotas until the outcome target is feasible."""
    failure_capacities = Counter(dataset_name(episode) for episode in failures)
    success_capacities = Counter(dataset_name(episode) for episode in successes)
    quotas = dict(total_quotas)

    def bounds() -> tuple[int, int]:
        minimum = sum(
            max(0, quota - failure_capacities[name])
            for name, quota in quotas.items()
        )
        maximum = sum(
            min(quota, success_capacities[name])
            for name, quota in quotas.items()
        )
        return minimum, maximum

    minimum_success, maximum_success = bounds()
    while minimum_success > target_success:
        donors = [
            name
            for name, quota in quotas.items()
            if quota > failure_capacities[name]
        ]
        recipients = [
            name
            for name, quota in quotas.items()
            if quota < failure_capacities[name]
            and quota
            < failure_capacities[name] + success_capacities[name]
        ]
        if not donors or not recipients:
            raise ValueError(
                f"Cannot lower the minimum success count from {minimum_success} "
                f"to {target_success}"
            )
        donor = min(donors, key=lambda name: (-quotas[name], name))
        recipient = min(recipients, key=lambda name: (quotas[name], name))
        quotas[donor] -= 1
        quotas[recipient] += 1
        minimum_success, maximum_success = bounds()

    while maximum_success < target_success:
        donors = [
            name
            for name, quota in quotas.items()
            if quota > success_capacities[name]
        ]
        recipients = [
            name
            for name, quota in quotas.items()
            if quota < success_capacities[name]
            and quota
            < failure_capacities[name] + success_capacities[name]
        ]
        if not donors or not recipients:
            raise ValueError(
                f"Cannot raise the maximum success count from {maximum_success} "
                f"to {target_success}"
            )
        donor = min(donors, key=lambda name: (-quotas[name], name))
        recipient = min(recipients, key=lambda name: (quotas[name], name))
        quotas[donor] -= 1
        quotas[recipient] += 1
        minimum_success, maximum_success = bounds()

    return quotas


def print_dataset_quotas(label: str, quotas: dict[str, int]) -> None:
    summary = ", ".join(f"{name}={count}" for name, count in sorted(quotas.items()))
    print(f"{label.capitalize()} dataset quotas: {summary or 'none'}", flush=True)


def pick_one_camera_per_episode(videos: list[dict], seed: int) -> list[dict]:
    by_episode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for video in videos:
        key = (str(video.get("dataset_repo") or ""), str(video.get("episode_id") or ""))
        by_episode[key].append(video)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for clips in by_episode.values():
        chosen.append(rng.choice(clips))
    return chosen


def instruction_from_info(info: str) -> str:
    text = html.unescape(info or "")
    # Generator writes: instruction<br><br>details
    parts = re.split(r"<br\s*/?>", text, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def public_info_html(instruction: str, camera_name: str, task_label: str) -> str:
    details = " / ".join([camera_name, task_label])
    return f"{html.escape(instruction)}<br><br>{html.escape(details)}"


def build_public_filters(videos: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    filters: dict[str, dict[str, str]] = {
        "camera": {},
        "task": {},
        "annotation": {},
        "failure_category": {},
        "severity": {},
    }
    for video in videos:
        filters["camera"][str(video["camera_id"])] = str(video["camera_label"])
        filters["task"][str(video["task_id"])] = str(video["task_label"])
        filters["annotation"][str(video["annotation_id"])] = str(video["annotation_label"])
        filters["severity"][str(video["severity_id"])] = str(video["severity_label"])
        for category_id, category_label in zip(
            video["failure_category_ids"],  # type: ignore[arg-type]
            video["failure_category_labels"],  # type: ignore[arg-type]
        ):
            filters["failure_category"][str(category_id)] = str(category_label)
    return filters


if __name__ == "__main__":
    main()
