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
from collections import defaultdict
from pathlib import Path

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

    sampled = rng.sample(failures, args.n_failure) + rng.sample(successes, args.n_success)
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
