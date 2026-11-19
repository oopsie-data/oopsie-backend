"""Check that every returned h5 belongs to the episode it claims to be.

A returned file ``pr_NNNN/<folder>/<batch>/episodes/NNNN.h5`` is accepted when

- its folder's manifest names the same annotator and lists the file's ``episode_id`` at its position,
- the batch's assignment gives that annotator the same episode at that position,
- the ``episode_id`` is the uid of exactly one episode index row, and
- the file's action and robot state arrays hash to that row's ``content_hash``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import polars as pl

from oopsie_backend.annotation.bundles import MANIFEST_NAME, sample_uid
from oopsie_backend.annotation.hub import Batch, PipelineConfig, folder_name, returned_batch_dirs
from oopsie_backend.common.h5 import _s
from oopsie_backend.index.core import episode_content_hash
from oopsie_backend.index.store import KEY


@dataclass
class FileCheck:
    path: Path
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


def assigned_positions(assignment_dir: Path) -> dict[str, dict[int, str]]:
    """Annotator folder -> position -> uid."""
    out: dict[str, dict[int, str]] = {}
    for path in sorted(assignment_dir.glob("*.json")):
        if path.name == "assignment_summary.json":
            continue
        lists = json.loads(path.read_text())
        out[folder_name(lists["annotator"])] = {
            e["position"]: sample_uid(e["dataset_repo"], e["repo_path"]) for e in lists["episodes"]
        }
    return out


def check_file(
    path: Path, folder: str, manifest: dict | None, assigned: dict[int, str] | None, index: dict[str, dict]
) -> FileCheck:
    check = FileCheck(path)
    with h5py.File(path, "r") as f:
        check.uid = _s(f.attrs.get("episode_id"))
        digest = episode_content_hash(f)
    position = int(path.stem)
    if manifest is None:
        check.problems.append("no manifest")
    else:
        if folder_name(manifest.get("annotator", "")) != folder:
            check.problems.append(f"manifest annotator {manifest.get('annotator')!r} is not folder {folder!r}")
        entry = next((e for e in manifest["episodes"] if e["bundle_path"] == f"episodes/{path.name}"), None)
        if entry is None:
            check.problems.append("not in manifest")
        elif entry["uid"] != check.uid or entry["position"] != position:
            check.problems.append(f"manifest has {entry['uid']} at position {entry['position']}")
    if assigned is None:
        check.problems.append(f"annotator folder {folder!r} not in the assignment")
    elif assigned.get(position) != check.uid:
        check.problems.append(f"assignment has {assigned.get(position)} at position {position}")
    row = index.get(check.uid)
    if row is None:
        check.problems.append("episode_id matches no index row")
    else:
        check.episode_key = row[KEY]
        if row["content_hash"] != digest:
            check.problems.append(f"data differs from index episode {row[KEY]}")
    return check


def verify_batch(cfg: PipelineConfig, batch: Batch, index: dict[str, dict]) -> list[FileCheck]:
    """Checks for every returned h5 of ``batch``."""
    assigned = assigned_positions(batch.assignment_dir)
    checks = []
    for batch_dir in returned_batch_dirs(cfg, batch.name):
        folder = batch_dir.parent.name
        manifest_path = batch_dir / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        for path in sorted((batch_dir / "episodes").glob("*.h5")):
            checks.append(check_file(path, folder, manifest, assigned.get(folder), index))
    return checks


def write_report(path: Path, checks: list[FileCheck]) -> None:
    rejected = [
        {"path": str(c.path), "uid": c.uid, "episode_key": c.episode_key, "problems": c.problems}
        for c in checks
        if c.problems
    ]
    path.write_text(json.dumps({"checked": len(checks), "rejected": rejected}, indent=2))
