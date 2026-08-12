#!/usr/bin/env python3
"""Generate website visualizer MP4s and metadata for OopsieData HDF5 episodes."""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import h5py

from oopsie_backend.utils.episode_stats import (
    HF_ORG,
    PUBLIC_EXCLUDED_DATASETS,
    canonical_dataset_repo,
    resolve_local_episode_path,
)


# =============================================================================
# Edit these constants on TACC.
# =============================================================================

DATASET_ROOT = Path("/media/arpit/2faaf81e-4247-445a-af59-ce2b3c62537c/oopsiedata/OopsieData-Submissions")
# Local Jekyll serves this path as /assets/visualizer/. Keep gitignored (large).
OUTPUT_ROOT = Path("/home/arpit/test_projects/oopsiedata/oopsie-docs/docs/assets/visualizer")
TOOLS_ROOT = Path("/home/arpit/test_projects/oopsiedata/oopsie_visualizer_tools")

METADATA_FILENAME = "metadata.json"
VIDEOS_DIRNAME = "videos"
POSTERS_DIRNAME = "posters"

# Set to a camera name such as "front_cam" to keep only that camera.
# Leave as None to use every camera listed in observations/video_paths/.
CAMERA_NAME: Optional[str] = None

FFMPEG = "ffmpeg"
FFMPEG_DOWNLOAD_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
VIDEO_FPS = 15
MAX_WIDTH = 640
MAX_HEIGHT = 360
CRF = 30
PRESET = "veryfast"
POSTER_TIME_SECONDS = 0.5

# Stop after this many HDF5 episodes. Leave as None for the full dataset.
MAX_EPISODES: Optional[int] = None


HDF5_SUFFIXES = {".h5", ".hdf5"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def main() -> None:
    ffmpeg = str(find_or_download_ffmpeg())
    (OUTPUT_ROOT / VIDEOS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / POSTERS_DIRNAME).mkdir(parents=True, exist_ok=True)

    videos = []
    processed_episodes = 0
    print(f"Using ffmpeg: {ffmpeg}", flush=True)
    print(f"Scanning {DATASET_ROOT}", flush=True)

    for h5_path in iter_hdf5_files():
        if MAX_EPISODES is not None and processed_episodes >= MAX_EPISODES:
            break
        processed_episodes += 1
        print(f"[{processed_episodes}] {h5_path}", flush=True)
        videos.extend(process_episode(h5_path, ffmpeg))

    metadata = {
        "videos": videos,
        "filters": build_filters(videos),
        "excluded_from_public_datasets": sorted(PUBLIC_EXCLUDED_DATASETS),
        "source": {"dataset_root": str(DATASET_ROOT)},
    }

    metadata_path = OUTPUT_ROOT / METADATA_FILENAME
    write_json_atomic(metadata_path, metadata)
    print(f"Wrote {len(videos)} videos to {metadata_path}")


def iter_hdf5_files():
    for dirpath, dirnames, filenames in os.walk(DATASET_ROOT):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in HDF5_SUFFIXES:
                yield path


def write_json_atomic(path: Path, data: object) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)


def find_or_download_ffmpeg() -> Path:
    configured_path = Path(FFMPEG)
    if configured_path.exists():
        return configured_path

    path_ffmpeg = shutil.which(FFMPEG)
    if path_ffmpeg is not None:
        return Path(path_ffmpeg)

    return download_ffmpeg()


def download_ffmpeg() -> Path:
    bin_dir = TOOLS_ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = bin_dir / "ffmpeg"
    if ffmpeg_path.exists():
        return ffmpeg_path

    archive_path = bin_dir / "ffmpeg-release-amd64-static.tar.xz"
    print(f"Downloading ffmpeg to {archive_path}")
    urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, archive_path)

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.isfile() and Path(member.name).name == "ffmpeg":
                member.name = "ffmpeg"
                archive.extract(member, bin_dir)
                ffmpeg_path.chmod(0o755)
                return ffmpeg_path

    raise RuntimeError(f"Downloaded archive did not contain an ffmpeg binary: {archive_path}")


def dataset_repo_for_path(h5_path: Path) -> str:
    """Path-derived HF repo id (same rules as internal stats)."""
    rel = h5_path.relative_to(DATASET_ROOT)
    resolved = resolve_local_episode_path(DATASET_ROOT, rel)
    if resolved is not None:
        return canonical_dataset_repo(resolved[0])
    # Fallback when the path is too shallow to resolve.
    leaf = rel.parts[0] if rel.parts else h5_path.stem
    return canonical_dataset_repo(f"{HF_ORG}/{leaf}" if DATASET_ROOT.name == HF_ORG else leaf)


def dataset_display_name(dataset_repo: str) -> str:
    """Short sidebar/card label: repo leaf (e.g. SOAR_origin)."""
    return dataset_repo.rsplit("/", 1)[-1]


def process_episode(h5_path: Path, ffmpeg: str) -> list[dict[str, object]]:
    dataset_repo = dataset_repo_for_path(h5_path)
    dataset_label = dataset_display_name(dataset_repo)
    bulk_import = dataset_repo in PUBLIC_EXCLUDED_DATASETS

    with h5py.File(h5_path, "r") as h5:
        language_instruction = read_attr(h5, "language_instruction", h5_path.stem)
        episode_id = read_attr(h5, "episode_id", h5_path.stem)
        annotation = read_failure_annotation(h5)
        success = annotation["success"]
        camera_paths = read_camera_paths(h5, h5_path)

    records = []
    for camera_name, source_video in camera_paths.items():
        output_video, output_poster = output_paths(h5_path, camera_name)
        try:
            transcode_video(source_video, output_video, ffmpeg)
        except subprocess.CalledProcessError:
            print(f"  retrying after removing partial output for {output_video}", flush=True)
            output_video.unlink(missing_ok=True)
            try:
                transcode_video(source_video, output_video, ffmpeg)
            except subprocess.CalledProcessError as error:
                print(f"  skipping {source_video}: ffmpeg failed with exit code {error.returncode}", flush=True)
                output_video.unlink(missing_ok=True)
                output_poster.unlink(missing_ok=True)
                continue

        try:
            make_poster(output_video, output_poster, ffmpeg)
        except subprocess.CalledProcessError:
            print(f"  retrying after removing partial poster for {output_poster}", flush=True)
            output_poster.unlink(missing_ok=True)
            try:
                make_poster(output_video, output_poster, ffmpeg)
            except subprocess.CalledProcessError as error:
                print(f"  keeping video without poster: ffmpeg failed with exit code {error.returncode}", flush=True)
                output_poster.unlink(missing_ok=True)

        task_label = success_label(success)
        records.append(
            {
                "src": output_video.relative_to(OUTPUT_ROOT).as_posix(),
                "poster": (
                    output_poster.relative_to(OUTPUT_ROOT).as_posix()
                    if output_poster.exists()
                    else ""
                ),
                "info": info_html(language_instruction, dataset_label, task_label, camera_name),
                "scene_id": filter_id(dataset_label, "scene"),
                "camera_id": filter_id(camera_name, "camera"),
                "camera_label": camera_name,
                "task_id": filter_id(task_label, "task"),
                "scene_label": dataset_label,
                "task_label": task_label,
                "dataset_repo": dataset_repo,
                "bulk_import": bulk_import,
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
                "episode_id": episode_id,
                "success": success,
            }
        )

    return records


def read_attr(h5: h5py.File, name: str, default: str) -> str:
    value = h5.attrs.get(name, default)
    return decode_text(value)


def read_failure_annotation(h5: h5py.File) -> dict[str, object]:
    annotation = {
        "success": None,
        "annotation_label": "No failure annotation",
        "failure_categories": ["No failure category"],
        "severity_label": "No severity",
        "failure_description": "",
        "additional_notes": "",
    }

    if "episode_annotations" not in h5:
        return annotation

    annotations = h5["episode_annotations"]
    for annotator_name in sorted(annotations.keys()):
        attrs = annotations[annotator_name].attrs
        taxonomy = parse_taxonomy(attrs.get("taxonomy"))
        failure_description = decode_text(attrs.get("failure_description", "")).strip()
        failure_categories = normalize_label_list(taxonomy.get("failure_category"))
        severity = decode_text(taxonomy.get("severity", "")).strip()
        additional_notes = decode_text(attrs.get("additional_notes", "")).strip()

        if "success" in attrs:
            annotation["success"] = float(attrs["success"])

        has_failure_annotation = bool(failure_description or failure_categories or severity)
        if has_failure_annotation:
            annotation.update(
                {
                    "annotation_label": "Has failure annotation",
                    "failure_categories": failure_categories or ["No failure category"],
                    "severity_label": severity or "No severity",
                    "failure_description": failure_description,
                    "additional_notes": additional_notes,
                }
            )
            return annotation

    return annotation


def parse_taxonomy(value: object) -> dict[str, object]:
    if value is None:
        return {}
    text = decode_text(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_label_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [decode_text(item).strip() for item in value if decode_text(item).strip()]
    text = decode_text(value).strip()
    if not text:
        return []
    return [text]


def read_camera_paths(h5: h5py.File, h5_path: Path) -> dict[str, Path]:
    if "observations/video_paths" not in h5:
        print(f"Skipping {h5_path}: no observations/video_paths group")
        return {}

    camera_paths = {}
    video_paths = h5["observations/video_paths"]
    for camera_name in sorted(video_paths.keys()):
        if CAMERA_NAME is not None and camera_name != CAMERA_NAME:
            continue

        source_video = Path(decode_text(video_paths[camera_name][()]))
        if not source_video.is_absolute():
            source_video = h5_path.parent / source_video

        if not source_video.exists() or source_video.suffix.lower() not in VIDEO_SUFFIXES:
            print(f"Skipping missing video: {source_video}")
            continue

        camera_paths[camera_name] = source_video

    return camera_paths


def output_paths(h5_path: Path, camera_name: str) -> tuple[Path, Path]:
    relative_h5 = h5_path.relative_to(DATASET_ROOT).with_suffix("")
    filename = f"{relative_h5.name}__{safe_name(camera_name)}"
    output_subdir = relative_h5.parent

    output_video = OUTPUT_ROOT / VIDEOS_DIRNAME / output_subdir / f"{filename}.mp4"
    output_poster = OUTPUT_ROOT / POSTERS_DIRNAME / output_subdir / f"{filename}.jpg"
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_poster.parent.mkdir(parents=True, exist_ok=True)
    return output_video, output_poster


def transcode_video(source_video: Path, output_video: Path, ffmpeg: str) -> None:
    if output_video.exists():
        return

    print(f"  transcoding {source_video} -> {output_video}", flush=True)
    temp_video = output_video.with_name(f"{output_video.stem}.tmp{output_video.suffix}")
    temp_video.unlink(missing_ok=True)
    scale = (
        f"scale='min({MAX_WIDTH},iw)':'min({MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-an",
            "-vf",
            (
                f"fps={VIDEO_FPS},{scale},"
                "pad='ceil(iw/2)*2':'ceil(ih/2)*2',format=yuv420p"
            ),
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            str(CRF),
            "-movflags",
            "+faststart",
            str(temp_video),
        ],
        check=True,
    )
    temp_video.replace(output_video)


def make_poster(source_video: Path, output_poster: Path, ffmpeg: str) -> None:
    if output_poster.exists():
        return

    print(f"  poster {output_poster}", flush=True)
    temp_poster = output_poster.with_name(f"{output_poster.stem}.tmp{output_poster.suffix}")
    temp_poster.unlink(missing_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(POSTER_TIME_SECONDS),
            "-i",
            str(source_video),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(temp_poster),
        ],
        check=True,
    )
    temp_poster.replace(output_poster)


def build_filters(videos: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    filters = {
        "scene": {},
        "camera": {},
        "task": {},
        "annotation": {},
        "failure_category": {},
        "severity": {},
    }
    for video in videos:
        filters["scene"][str(video["scene_id"])] = str(video["scene_label"])
        filters["camera"][str(video["camera_id"])] = str(video["camera_label"])
        filters["task"][str(video["task_id"])] = str(video["task_label"])
        filters["annotation"][str(video["annotation_id"])] = str(video["annotation_label"])
        filters["severity"][str(video["severity_id"])] = str(video["severity_label"])
        for category_id, category_label in zip(
            video["failure_category_ids"],
            video["failure_category_labels"],
        ):
            filters["failure_category"][str(category_id)] = str(category_label)
    return filters


def info_html(instruction: str, dataset_label: str, task_label: str, camera_name: str) -> str:
    details = " / ".join([dataset_label, camera_name, task_label])
    return f"{html.escape(instruction)}<br><br>{html.escape(details)}"


def success_label(success: Optional[float]) -> str:
    if success is None:
        return "Unknown"
    return "Success" if success >= 0.5 else "Failure"


def filter_id(label: str, prefix: str) -> str:
    return f"{prefix}-{safe_name(label).lower()}"


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "unknown"


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


if __name__ == "__main__":
    main()
