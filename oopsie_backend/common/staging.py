"""Copy episodes from the mirror to a staging directory on fast storage before reading them.

``stage_episodes`` copies each episode's h5 and videos to ``<stage_dir>``,
under the same layout as the mirror, so ``episode_path(stage_dir, ...)`` finds
the copy and the copy finds its videos. A file is copied only when the staged
copy is missing or differs in size or modification time, so the staging
directory works as a cache across runs.

The whole dataset never fits on the staging storage, so a call stages at most
``STAGE_CHUNK`` episodes and raises for more. Callers stage in chunks and
remove each chunk's copies with ``unstage_episodes`` once they are done with it.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oopsie_backend.common.repos import episode_path
from oopsie_backend.common.videos import _video_files, repo_root

STAGE_CHUNK = 1000


def chunks(items: list, size: int = STAGE_CHUNK) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _copy(src: Path, dst: Path) -> int:
    st = src.stat()
    if dst.is_file():
        dt = dst.stat()
        if dt.st_size == st.st_size and dt.st_mtime_ns == st.st_mtime_ns:
            return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return st.st_size


def stage_episode(data_dir: Path, stage_dir: Path, dataset_repo: str, repo_path: str) -> int:
    """Stage one episode; returns the number of bytes copied."""
    src = episode_path(data_dir, dataset_repo, repo_path)
    dst = episode_path(stage_dir, dataset_repo, repo_path)
    if not src.is_file():
        return 0
    copied = _copy(src, dst)
    src_root, dst_root = repo_root(src, repo_path), repo_root(dst, repo_path)
    for video, rel in _video_files(src, src_root).values():
        copied += _copy(video, dst_root / rel)
    return copied


def unstage_episode(data_dir: Path, stage_dir: Path, dataset_repo: str, repo_path: str) -> None:
    """Remove the staged copy of one episode and its videos."""
    src = episode_path(data_dir, dataset_repo, repo_path)
    dst = episode_path(stage_dir, dataset_repo, repo_path)
    if src.is_file():
        dst_root = repo_root(dst, repo_path)
        for _, rel in _video_files(src, repo_root(src, repo_path)).values():
            (dst_root / rel).unlink(missing_ok=True)
    dst.unlink(missing_ok=True)


def unstage_episodes(data_dir: Path, stage_dir: Path, episodes: list[tuple[str, str]]) -> None:
    for e in episodes:
        unstage_episode(data_dir, stage_dir, *e)


def stage_episodes(
    data_dir: Path,
    stage_dir: Path,
    episodes: list[tuple[str, str]],
    workers: int = 8,
    stop: Callable[[], bool] | None = None,
) -> Path:
    """Stage at most ``STAGE_CHUNK`` ``(dataset_repo, repo_path)`` episodes; returns ``stage_dir`` for
    use as the data dir. Once ``stop`` returns true, the remaining episodes are not copied."""
    if len(episodes) > STAGE_CHUNK:
        raise ValueError(f"refusing to stage {len(episodes)} episodes at once; stage at most {STAGE_CHUNK} per call")

    def one(e: tuple[str, str]) -> int:
        return 0 if stop is not None and stop() else stage_episode(data_dir, stage_dir, *e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        copied = sum(pool.map(one, episodes))
    print(f"staged {len(episodes)} episode(s) in {stage_dir}, copied {copied / 1e9:.2f} GB", file=sys.stderr)
    return stage_dir
