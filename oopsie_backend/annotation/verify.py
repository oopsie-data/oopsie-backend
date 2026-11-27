"""Check that every returned h5 belongs to the episode it claims to be.

A returned file ``pr_NNNN/<folder>/<slot>/episodes/NNNN.h5`` is accepted when

- the roster gives or gave that slot of that folder a bundle, and the bundle has the file's ``episode_id`` at
  its position,
- the ``episode_id`` is the uid of exactly one episode index row, and
- the file's action and robot state arrays hash to that row's ``content_hash``, or to the content hash of the
  same uid in one of the previous indexes passed with ``--previous-index`` (episodes re-uploaded after their
  bundle was built).
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
from oopsie_backend.common.h5 import _s, is_h5
from oopsie_backend.index.core import episode_content_hash
from oopsie_backend.index.store import KEY


@dataclass
class FileCheck:
    path: Path
    annotator: str | None = None
    kind: str | None = None
    uid: str | None = None
    episode_key: str | None = None
    problems: list[str] = field(default_factory=list)


def index_by_uid(index: Path, columns: tuple[str, ...] = ()) -> dict[str, dict]:
    """uid -> ``episode_key``, ``dataset_repo``, ``repo_path``, ``content_hash`` and ``columns`` of every index row."""
    base = [KEY, "dataset_repo", "repo_path", "content_hash"]
    df = pl.read_parquet(index, columns=base + [c for c in columns if c not in base])
    out: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        uid = sample_uid(row["dataset_repo"], row["repo_path"])
        if uid in out:
            raise ValueError(f"uid {uid} matches {out[uid][KEY]} and {row[KEY]}")
        out[uid] = row
    return out


def accepted_hashes(index: dict[str, dict], previous: list[Path]) -> dict[str, set[str]]:
    """uid -> content hashes of the uid in ``index`` and in each ``previous`` index."""
    out = {uid: {row["content_hash"]} for uid, row in index.items()}
    for path in previous:
        for uid, row in index_by_uid(path).items():
            if uid in out and row["content_hash"]:
                out[uid].add(row["content_hash"])
    return out


def check_file(
    path: Path,
    positions: dict[int, str] | None,
    index: dict[str, dict],
    hashes: dict[str, set[str]] | None = None,
) -> FileCheck:
    check = FileCheck(path)
    with h5py.File(path, "r") as f:
        check.uid = _s(f.attrs.get("episode_id"))
        digest = episode_content_hash(f)
    position = int(path.stem)
    if positions is None:
        check.problems.append("slot is not in the roster")
    elif positions.get(position) != check.uid:
        check.problems.append(f"bundle has {positions.get(position)} at position {position}")
    row = index.get(check.uid)
    if row is None:
        check.problems.append("episode_id matches no index row")
    else:
        check.episode_key = row[KEY]
        if digest not in (hashes or {}).get(check.uid, {row["content_hash"]}):
            check.problems.append(f"data differs from index episode {row[KEY]}")
    return check


def verify_returns(
    cfg: PipelineConfig, registry: Registry, index: dict[str, dict], hashes: dict[str, set[str]] | None = None
) -> list[FileCheck]:
    """Checks for every returned h5, with the annotator and bundle kind of its slot."""
    slots = registry.by_slot()
    checks = []
    for slot_dir in returned_slot_dirs(cfg):
        holding = slots.get((slot_dir.parent.name, slot_dir.name))
        positions = registry.positions(holding) if holding else None
        for path in sorted(p for p in (slot_dir / "episodes").glob("*") if is_h5(p)):
            check = check_file(path, positions, index, hashes)
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
    path.write_text(json.dumps({"checked": len(checks), "rejected": rejected}, indent=2))
