"""Check that every returned h5 belongs to the episode it claims to be, and that the episode still shows the same videos.

A returned file ``pr_NNNN/<folder>/<slot>/episodes/NNNN.h5`` is rejected unless

- the roster gives or gave that slot of that folder a bundle, and the bundle has the file's ``episode_id`` at
  its position, and
- the ``episode_id`` is the uid of exactly one episode index row.

A file that passes is unmatched when a video the annotator watched (``annotation.videos``) differs from the
index episode's video of the same camera, or when the bundle's videos cannot be found. Unmatched files are
reported and left out; the check runs again on every ``verify`` and ``collect``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import polars as pl

from oopsie_backend.annotation.bundles import sample_uid
from oopsie_backend.annotation.hub import PipelineConfig, returned_slot_dirs
from oopsie_backend.annotation.roster import Registry
from oopsie_backend.annotation.videos import (
    CACHE_NAME,
    bundle_videos,
    videos_match,
    watched,
)
from oopsie_backend.common.h5 import _s, is_h5
from oopsie_backend.index.store import KEY


@dataclass
class FileCheck:
    path: Path
    annotator: str | None = None
    kind: str | None = None
    uid: str | None = None
    episode_key: str | None = None
    problems: list[str] = field(default_factory=list)
    unmatched: str | None = None
    seen: dict[str, str] | None = None


def index_by_uid(index: Path, columns: tuple[str, ...] = ()) -> dict[str, dict]:
    """uid -> ``episode_key``, ``dataset_repo``, ``repo_path``, ``videos`` and ``columns`` of every index row."""
    base = [KEY, "dataset_repo", "repo_path", "videos"]
    df = pl.read_parquet(index, columns=base + [c for c in columns if c not in base])
    out: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        uid = sample_uid(row["dataset_repo"], row["repo_path"])
        if uid in out:
            raise ValueError(f"uid {uid} matches {out[uid][KEY]} and {row[KEY]}")
        out[uid] = row
    return out


def check_file(
    path: Path,
    positions: dict[int, str] | None,
    index: dict[str, dict],
    seen: dict[str, str] | None,
) -> FileCheck:
    check = FileCheck(path, seen=seen)
    with h5py.File(path, "r") as f:
        check.uid = _s(f.attrs.get("episode_id"))
    position = int(path.stem)
    if positions is None:
        check.problems.append("slot is not in the roster")
    elif positions.get(position) != check.uid:
        check.problems.append(f"bundle has {positions.get(position)} at position {position}")
    row = index.get(check.uid)
    if row is None:
        check.problems.append("episode_id matches no index row")
        return check
    check.episode_key = row[KEY]
    if not seen:
        check.unmatched = "the bundle's videos are not in the batch repo"
    elif not videos_match(seen, row["videos"]):
        check.unmatched = f"the videos differ from those of index episode {row[KEY]}"
    return check


def verify_returns(cfg: PipelineConfig, registry: Registry, index: dict[str, dict]) -> list[FileCheck]:
    """Checks for every returned h5, with the annotator and bundle kind of its slot."""
    slots = registry.by_slot()
    slot_dirs = returned_slot_dirs(cfg)
    videos = bundle_videos(cfg.repo_id, {(d.parent.name, d.name) for d in slot_dirs}, cfg.returns_dir / CACHE_NAME)
    checks = []
    for slot_dir in slot_dirs:
        holding = slots.get((slot_dir.parent.name, slot_dir.name))
        positions = registry.positions(holding) if holding else None
        for path in sorted(p for p in (slot_dir / "episodes").glob("*") if is_h5(p)):
            check = check_file(path, positions, index, watched(videos, path))
            if holding:
                check.annotator = holding.annotator
                check.kind = registry.bundles[holding.bundle]["kind"]
            checks.append(check)
    return checks


def write_report(path: Path, checks: list[FileCheck]) -> None:
    rejected = [
        {"path": str(c.path), "uid": c.uid, "episode_key": c.episode_key, "problems": c.problems}
        for c in checks
        if c.problems
    ]
    unmatched = [
        {"path": str(c.path), "uid": c.uid, "episode_key": c.episode_key, "reason": c.unmatched}
        for c in checks
        if c.unmatched and not c.problems
    ]
    path.write_text(json.dumps({"checked": len(checks), "rejected": rejected, "unmatched": unmatched}, indent=2))
