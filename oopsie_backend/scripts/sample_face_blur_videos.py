r"""Create a small hand-labeling set of videos for face-blur evaluation.

Examples:
    # From an already-synced local dataset repo.
    uv run python -m oopsie_backend.scripts.sample_face_blur_videos \
        --local-dir /corral/utexas/IRI26024/oopsiedata/OopsieData-Submissions/UofT_TISL \
        --output-dir /datastor1/jiahuikchen/face_blur_testset \
        --count 80

    # Directly from Hugging Face, without syncing the full repo.
    uv run python -m oopsie_backend.scripts.sample_face_blur_videos \
        --repo OopsieData-Submissions/UofT_TISL \
        --output-dir /datastor1/jiahuikchen/face_blur_testset \
        --count 80
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from oopsie_backend.utils.hf import list_repo_files, parse_repo_arg

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
MANIFEST_FIELDNAMES = [
    "sample_id",
    "source_kind",
    "source_repo",
    "source_path",
    "local_video_path",
    "face_present",
    "labeler",
    "notes",
]


def _is_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    safe = safe.strip("._")
    return safe or "video"


def _sample(items: list[str], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    items = sorted(items)
    if count >= len(items):
        return items
    return sorted(rng.sample(items, count))


def _read_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        missing = [name for name in MANIFEST_FIELDNAMES if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"existing manifest is missing column(s): {', '.join(missing)}")
        return [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in reader
        ]


def _next_sample_number(rows: list[dict[str, str]]) -> int:
    next_number = 1
    for row in rows:
        match = re.fullmatch(r"video_(\d+)", row.get("sample_id", ""))
        if match:
            next_number = max(next_number, int(match.group(1)) + 1)
    return next_number


def _existing_source_paths(
    rows: list[dict[str, str]],
    *,
    source_kind: str,
    source_repo: str,
) -> set[str]:
    return {
        row.get("source_path", "")
        for row in rows
        if row.get("source_kind") == source_kind
        and row.get("source_repo", "") == source_repo
        and row.get("source_path")
    }


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _write_labeling_readme(output_dir: Path, manifest_name: str) -> None:
    text = f"""# Face-Blur Hand-Labeling Set

Fill in `{manifest_name}` after reviewing each video.

Label values:

- `yes`: at least one visible human face appears in the video.
- `no`: no visible human face appears.
- `uncertain`: labeler cannot confidently decide.

Keep `sample_id`, `source_kind`, `source_repo`, `source_path`, and
`local_video_path` unchanged. Use `notes` for edge cases such as reflections,
posters, extreme blur, partial faces, or faces visible for only a few frames.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def sample_from_local(
    local_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    copy: bool,
    start_number: int = 1,
    exclude_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"local dir not found: {local_dir}")

    videos = [
        p.relative_to(local_dir).as_posix()
        for p in local_dir.rglob("*")
        if p.is_file() and _is_video(p.name) and ".cache" not in p.parts
    ]
    if exclude_paths:
        videos = [path for path in videos if path not in exclude_paths]
    selected = _sample(videos, count, seed)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for offset, rel_path in enumerate(selected):
        src = local_dir / rel_path
        sample_id = f"video_{start_number + offset:04d}"
        dst_name = f"{sample_id}__{_safe_name(Path(rel_path).name)}"
        dst = videos_dir / dst_name
        _link_or_copy(src, dst, copy=copy)
        rows.append(
            {
                "sample_id": sample_id,
                "source_kind": "local",
                "source_repo": "",
                "source_path": rel_path,
                "local_video_path": dst.relative_to(output_dir).as_posix(),
                "face_present": "",
                "labeler": "",
                "notes": "",
            }
        )
    return rows


def sample_from_hf(
    repo_arg: str,
    output_dir: Path,
    *,
    repo_type: str,
    token: str | None,
    count: int,
    seed: int,
    start_number: int = 1,
    exclude_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    repo_id, repo_type = parse_repo_arg(repo_arg, repo_type)
    files = list_repo_files(repo_id, token, repo_type)
    videos = [p for p in files if _is_video(p)]
    if exclude_paths:
        videos = [path for path in videos if path not in exclude_paths]
    selected = _sample(videos, count, seed)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for offset, repo_path in enumerate(selected):
        sample_id = f"video_{start_number + offset:04d}"
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type=repo_type,
                token=token,
            )
        )
        dst_name = f"{sample_id}__{_safe_name(Path(repo_path).name)}"
        dst = videos_dir / dst_name
        shutil.copy2(cached, dst)
        rows.append(
            {
                "sample_id": sample_id,
                "source_kind": "huggingface",
                "source_repo": repo_id,
                "source_path": repo_path,
                "local_video_path": dst.relative_to(output_dir).as_posix(),
                "face_present": "",
                "labeler": "",
                "notes": "",
            }
        )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_labeling_readme(output_dir, manifest.name)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--local-dir",
        type=Path,
        help="Local dataset repo directory containing videos.",
    )
    source.add_argument(
        "--repo",
        help="Hugging Face dataset id or URL. Downloads only sampled videos.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="Hugging Face repo type for --repo.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token; defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write videos/, manifest.csv, and README.md.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=80,
        help="Number of videos to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260714,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy local videos instead of symlinking them. HF samples are always copied.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append sampled videos to an existing manifest.csv in --output-dir.",
    )
    args = parser.parse_args(argv)

    if args.count <= 0:
        print("error: --count must be positive", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"

    try:
        existing_rows = _read_existing_manifest(manifest) if args.append else []
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start_number = _next_sample_number(existing_rows)

    try:
        if args.local_dir is not None:
            exclude_paths = _existing_source_paths(
                existing_rows,
                source_kind="local",
                source_repo="",
            )
            rows = sample_from_local(
                args.local_dir.resolve(),
                output_dir,
                count=args.count,
                seed=args.seed,
                copy=args.copy,
                start_number=start_number,
                exclude_paths=exclude_paths,
            )
        else:
            assert args.repo is not None
            repo_id, _ = parse_repo_arg(args.repo, args.repo_type)
            exclude_paths = _existing_source_paths(
                existing_rows,
                source_kind="huggingface",
                source_repo=repo_id,
            )
            rows = sample_from_hf(
                args.repo,
                output_dir,
                repo_type=args.repo_type,
                token=args.token,
                count=args.count,
                seed=args.seed,
                start_number=start_number,
                exclude_paths=exclude_paths,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("error: no video files found in source", file=sys.stderr)
        return 1

    manifest = write_manifest(output_dir, [*existing_rows, *rows])
    print(f"Wrote {len(rows)} sampled video(s)")
    print(f"Manifest: {manifest}")
    print(f"Videos:   {output_dir / 'videos'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
