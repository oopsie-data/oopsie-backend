"""Locate the video files of an episode next to its h5."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import h5py

from oopsie_backend.common.h5 import _s


def _video_files(h5_path: Path, repo_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map camera -> ``(source_file, repo_relative_bundle_path)``.

    The bundle path is where the annotator will look first for the stored
    ``video_paths`` entry, so the copied h5 resolves its videos unchanged.
    Falls back to ``<stem>_<cam>.mp4`` next to the h5.
    """
    repo_dir = repo_dir.resolve()
    h5_dir = h5_path.parent.resolve()
    out: dict[str, tuple[Path, str]] = {}
    with h5py.File(h5_path, "r") as f:
        group = f.get("observations/video_paths")
        raw_paths = (
            {str(cam): _s(group[cam][()]) for cam in group}
            if isinstance(group, h5py.Group)
            else {}
        )
    for cam, raw in raw_paths.items():
        if not raw:
            continue
        candidate = PurePosixPath(raw)
        target = None
        if not candidate.is_absolute():
            target = (h5_dir / candidate).resolve()
            if not target.is_relative_to(repo_dir):
                target = None
        if target is None:
            target = h5_dir / candidate.name
        source = target if target.is_file() else h5_dir / candidate.name
        if not source.is_file():
            continue
        out[cam] = (source, target.relative_to(repo_dir).as_posix())
    if not out:
        stem = h5_path.stem
        for mp4 in sorted(h5_path.parent.glob(f"{stem}_*.mp4")):
            cam = mp4.stem[len(stem) + 1 :]
            if cam:
                out[cam] = (mp4, mp4.resolve().relative_to(repo_dir).as_posix())
    return out


def repo_root(path: Path, repo_path: str) -> Path:
    """The HF repo directory containing ``path``, whose path inside the repo is ``repo_path``."""
    root = path.resolve()
    for _ in PurePosixPath(repo_path).parts:
        root = root.parent
    return root


def episode_videos(path: Path, repo_path: str) -> dict[str, Path]:
    """Map camera -> mp4 file of the episode at ``path``."""
    return {cam: src for cam, (src, _) in _video_files(path, repo_root(path, repo_path)).items()}
