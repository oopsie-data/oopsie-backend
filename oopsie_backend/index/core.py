"""Core columns of the episode index: h5 metadata, file references and Hub hashes.

Episodes are discovered under ``<data_dir>/<HF_ORG>/<repo>/`` and, for the
local imports of ``configs/imports.yaml``, ``<data_dir>/imports/<name>/``.
Archive and infra repos (``configs/repositories.yaml``) are not indexed;
external datasets are indexed as ``external`` with ``public = false``. A file is re-read
only when its size or modification time changed. File hashes come from the Hub
listing: ``sha256`` for LFS files and the git object id for all files.
``content_hash`` hashes the action and robot state arrays, so copies of an
episode share it; ``preqa.duplicate_episodes`` uses it to mark duplicates.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from oopsie_data_tools.annotation_tool.annotation_schema import read_annotation_attrs

from oopsie_backend.common.h5 import _n_timesteps, _profile, _s, _str_list
from oopsie_backend.common.hf import hub_files
from oopsie_backend.common.repos import (
    CONTRIBUTOR,
    EPISODE_KINDS,
    HF_ORG,
    PUBLIC_EXCLUDED_LABS,
    data_root,
    import_dataset_id,
    is_local_import,
    load_imports,
    org_dir,
    repo_kind,
)
from oopsie_backend.common.videos import _video_files
from oopsie_backend.index.store import KEY, episode_key

IGNORED_DIRS = {".cache"}
GRIPPER_ACTIONS = ("gripper_position", "gripper_velocity", "gripper_binary")
MIN_TIMESTAMP = datetime(2000, 1, 1, tzinfo=timezone.utc)

FILE_STRUCT = pl.Struct(
    {"camera": pl.String, "path": pl.String, "size": pl.Int64, "sha256": pl.String, "git_oid": pl.String}
)
ANNOTATION_STRUCT = pl.Struct(
    {
        "name": pl.String,
        "timestamp": pl.String,
        "success": pl.Float64,
        "outcome": pl.String,
        "failure_category": pl.List(pl.String),
        "severity": pl.String,
        "episode_description": pl.String,
        "additional_notes": pl.String,
    }
)
CORE_SCHEMA = {
    KEY: pl.String,
    "dataset_repo": pl.String,
    "repo_path": pl.String,
    "repo_kind": pl.String,
    "public": pl.Boolean,
    "on_disk": pl.Boolean,
    "on_hub": pl.Boolean,
    "schema": pl.String,
    "episode_id": pl.String,
    "lab_id": pl.String,
    "operator_name": pl.String,
    "language_instruction": pl.String,
    "task": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "robot_name": pl.String,
    "gripper_name": pl.String,
    "policy_name": pl.String,
    "control_freq": pl.Float64,
    "is_biarm": pl.Boolean,
    "uses_mobile_base": pl.Boolean,
    "single_arm": pl.Boolean,
    "biarm": pl.Boolean,
    "gripper_dof": pl.Int64,
    "binary_gripper": pl.Boolean,
    "dexterous_hand": pl.Boolean,
    "action_space": pl.List(pl.String),
    "robot_state_keys": pl.List(pl.String),
    "cameras": pl.List(pl.String),
    "num_steps": pl.Int64,
    "duration_seconds": pl.Float64,
    "annotations": pl.List(ANNOTATION_STRUCT),
    "content_hash": pl.String,
    "h5_size": pl.Int64,
    "h5_mtime_ns": pl.Int64,
    "h5_sha256": pl.String,
    "h5_git_oid": pl.String,
    "videos": pl.List(FILE_STRUCT),
    "read_error": pl.String,
    "indexed_at": pl.Datetime("us", "UTC"),
}
# Columns derived from the h5 contents; reused while the file is unchanged.
H5_COLUMNS = [
    "schema", "episode_id", "lab_id", "operator_name", "language_instruction", "task", "timestamp",
    "robot_name", "gripper_name", "policy_name", "control_freq", "is_biarm", "uses_mobile_base",
    "single_arm", "biarm", "gripper_dof", "binary_gripper", "dexterous_hand",
    "action_space", "robot_state_keys", "cameras", "num_steps", "duration_seconds", "annotations",
    "content_hash", "videos", "read_error", "indexed_at",
]


def normalize_task(instruction: str | None) -> str:
    """Lowercase, collapse whitespace and strip surrounding punctuation."""
    text = re.sub(r"\s+", " ", (instruction or "").strip().lower())
    return text.strip(" .!?,;:\"'")


def discover(data_dir: Path) -> list[tuple[str, str, Path, str]]:
    """``(dataset_repo, repo_path, path, repo_kind)`` for every h5 in an indexed repo."""
    repos = [
        (f"{HF_ORG}/{p.name}", p)
        for p in sorted(org_dir(data_dir).iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    for name, entry in load_imports().items():
        dataset_repo = import_dataset_id(name, entry)
        if is_local_import(dataset_repo) and (data_root(data_dir) / dataset_repo).is_dir():
            repos.append((dataset_repo, data_root(data_dir) / dataset_repo))
    out = []
    for dataset_repo, repo_dir in repos:
        kind = repo_kind(dataset_repo)
        if kind not in EPISODE_KINDS:
            continue
        for path in sorted(repo_dir.rglob("*.h5")):
            rel = path.relative_to(repo_dir)
            if IGNORED_DIRS.intersection(rel.parts):
                continue
            out.append((dataset_repo, rel.as_posix(), path, kind))
    return out


def _timestamp(raw) -> datetime | None:
    try:
        dt = datetime.fromtimestamp(float(np.asarray(raw).item()), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return dt if dt >= MIN_TIMESTAMP else None


def _content_hash(f: h5py.File) -> str:
    h = hashlib.sha256()
    for group_name in ("actions", "observations/robot_states"):
        group = f.get(group_name)
        if not isinstance(group, h5py.Group):
            continue
        for name in sorted(group):
            ds = group[name]
            if not isinstance(ds, h5py.Dataset) or ds.shape is None:
                continue
            data = np.ascontiguousarray(ds[()])
            h.update(f"{group_name}/{name}:{data.dtype.str}:{data.shape};".encode())
            h.update(data.tobytes())
    return h.hexdigest()


def _width(group: h5py.Group | None, key: str) -> int | None:
    ds = group.get(key) if group is not None else None
    if not isinstance(ds, h5py.Dataset) or ds.shape is None or not ds.size:
        return None
    return int(np.prod(ds.shape[1:])) if len(ds.shape) > 1 else 1


def embodiment(actions: h5py.Group | None, is_biarm: bool | None) -> dict:
    """Arm and end-effector flags from the profile's ``is_biarm`` and the recorded gripper actions.

    ``gripper_dof`` is the width of the gripper action (``gripper_position``, else
    ``gripper_velocity``, else ``gripper_binary``) per arm. ``binary_gripper`` holds when the
    episode records a ``gripper_binary`` action or has one gripper value per arm;
    ``dexterous_hand`` when it has more than one.
    """
    arms = None if is_biarm is None else 2 if is_biarm else 1
    width = next((w for k in GRIPPER_ACTIONS if (w := _width(actions, k)) is not None), None)
    dof = width // arms if width is not None and arms and width % arms == 0 else None
    has_binary = _width(actions, "gripper_binary") is not None
    return {
        "single_arm": None if arms is None else arms == 1,
        "biarm": None if arms is None else arms == 2,
        "gripper_dof": dof,
        "binary_gripper": (has_binary or dof == 1) if (has_binary or dof is not None) else None,
        "dexterous_hand": dof > 1 if dof is not None else None,
    }


def _annotations(f: h5py.File) -> list[dict]:
    group = f.get("episode_annotations")
    if not isinstance(group, h5py.Group):
        return []
    out = []
    for name, g in group.items():
        if not isinstance(g, h5py.Group):
            continue
        decoded = read_annotation_attrs(g.attrs)
        success = g.attrs.get("success")
        out.append(
            {
                "name": str(name),
                "timestamp": _s(g.attrs.get("timestamp")),
                "success": float(np.asarray(success).item()) if success is not None else None,
                "outcome": decoded.get("outcome"),
                "failure_category": decoded.get("failure_category"),
                "severity": decoded.get("severity"),
                "episode_description": decoded.get("episode_description"),
                "additional_notes": decoded.get("additional_notes"),
            }
        )
    return out


def read_h5(path: Path, repo_path: str) -> dict:
    """The ``H5_COLUMNS`` of one episode; on failure only ``read_error`` is set."""
    now = datetime.now(timezone.utc)
    try:
        with h5py.File(path, "r") as f:
            profile = _profile(_s(f.attrs.get("robot_profile")))
            actions = f.get("actions") if isinstance(f.get("actions"), h5py.Group) else None
            video_group = f.get("observations/video_paths")
            try:
                control_freq = float(profile["control_freq"]) if profile.get("control_freq") else None
            except (TypeError, ValueError):
                control_freq = None
            n = _n_timesteps(actions)
            instruction = _s(f.attrs.get("language_instruction"))
            is_biarm = profile.get("is_biarm") if isinstance(profile.get("is_biarm"), bool) else None
            row = {
                "schema": _s(f.attrs.get("schema")),
                "episode_id": _s(f.attrs.get("episode_id")),
                "lab_id": _s(f.attrs.get("lab_id")),
                "operator_name": _s(f.attrs.get("operator_name")),
                "language_instruction": instruction,
                "task": normalize_task(instruction),
                "timestamp": _timestamp(f.attrs.get("timestamp")),
                "robot_name": _s(profile.get("robot_name")),
                "gripper_name": _s(profile.get("gripper_name")),
                "policy_name": _s(profile.get("policy_name")),
                "control_freq": control_freq,
                "is_biarm": is_biarm,
                "uses_mobile_base": profile.get("uses_mobile_base")
                if isinstance(profile.get("uses_mobile_base"), bool)
                else None,
                **embodiment(actions, is_biarm),
                "action_space": _str_list(profile.get("action_space")),
                "robot_state_keys": _str_list(profile.get("robot_state_keys")),
                "cameras": sorted(video_group.keys()) if isinstance(video_group, h5py.Group) else [],
                "num_steps": n,
                "duration_seconds": n / control_freq if n and control_freq else None,
                "annotations": _annotations(f),
                "content_hash": _content_hash(f),
            }
        repo_dir = path
        for _ in Path(repo_path).parts:
            repo_dir = repo_dir.parent
        row["videos"] = [
            {"camera": cam, "path": rel, "size": src.stat().st_size, "sha256": None, "git_oid": None}
            for cam, (src, rel) in sorted(_video_files(path, repo_dir).items())
        ]
        row["read_error"] = None
    except Exception as exc:
        row = {c: None for c in H5_COLUMNS} | {"read_error": f"{type(exc).__name__}: {exc}"}
    row["indexed_at"] = now
    return row


def _read_job(job: tuple[str, Path]) -> dict:
    repo_path, path = job
    return read_h5(path, repo_path)


def read_jobs(jobs: list[tuple[str, Path]], workers: int) -> list[dict]:
    """``read_h5`` of every ``(repo_path, path)``, in order."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_read_job, jobs, chunksize=32))


def build_core(
    data_dir: Path,
    previous: pl.DataFrame | None,
    api=None,
    workers: int = 8,
    read: Callable[[list[tuple[str, Path]], int], list[dict]] = read_jobs,
) -> pl.DataFrame:
    """Core columns for every discovered episode, plus earlier rows no longer on disk.
    ``read`` reads the new or changed h5 files, by default on this node."""
    found = discover(data_dir)
    prev = {r[KEY]: r for r in previous.iter_rows(named=True)} if previous is not None else {}
    # Rows from an index without every h5 column are read again, so new columns get filled.
    reuse = previous is not None and set(H5_COLUMNS) <= set(previous.columns)
    rows: dict[str, dict] = {}
    to_read: list[tuple[str, tuple[str, Path]]] = []
    for dataset_repo, repo_path, path, kind in found:
        key = episode_key(dataset_repo, repo_path)
        st = path.stat()
        row = {
            KEY: key,
            "dataset_repo": dataset_repo,
            "repo_path": repo_path,
            "repo_kind": kind,
            "on_disk": True,
            "h5_size": st.st_size,
            "h5_mtime_ns": st.st_mtime_ns,
        }
        old = prev.get(key)
        if reuse and old and old.get("h5_size") == st.st_size and old.get("h5_mtime_ns") == st.st_mtime_ns and not old.get("read_error"):
            row |= {c: old.get(c) for c in H5_COLUMNS}
        else:
            to_read.append((key, (repo_path, path)))
        rows[key] = row

    print(f"{len(found)} episode(s) found, {len(to_read)} new or changed", file=sys.stderr)
    for (key, _), result in zip(to_read, read([j for _, j in to_read], workers)):
        rows[key] |= result

    for key, old in prev.items():
        if key not in rows:
            rows[key] = {c: old.get(c) for c in CORE_SCHEMA} | {"on_disk": False}

    hub: dict[str, dict[str, dict] | None] = {}
    if api is not None:
        for repo in sorted({r["dataset_repo"] for r in rows.values()}):
            if is_local_import(repo):
                hub[repo] = {}
                continue
            try:
                hub[repo] = hub_files(api, repo)
            except Exception as exc:
                print(f"  hub listing failed for {repo}: {exc}", file=sys.stderr)
                hub[repo] = None
    for row in rows.values():
        listing = hub.get(row["dataset_repo"])
        old = prev.get(row[KEY], {})
        if listing is None:
            for c in ("on_hub", "h5_sha256", "h5_git_oid"):
                row[c] = old.get(c)
            continue
        entry = listing.get(row["repo_path"])
        row["on_hub"] = entry is not None
        row["h5_sha256"] = entry["sha256"] if entry else None
        row["h5_git_oid"] = entry["git_oid"] if entry else None
        for video in row.get("videos") or []:
            v = listing.get(video["path"])
            video["sha256"] = v["sha256"] if v else None
            video["git_oid"] = v["git_oid"] if v else None

    for row in rows.values():
        lab = row.get("lab_id")
        row["public"] = row["repo_kind"] == CONTRIBUTOR and lab not in PUBLIC_EXCLUDED_LABS
    df = pl.DataFrame(
        [{c: row.get(c) for c in CORE_SCHEMA} for row in rows.values()], schema=CORE_SCHEMA
    )
    return df.select(list(CORE_SCHEMA))


def merge_core(core: pl.DataFrame, existing: pl.DataFrame | None) -> pl.DataFrame:
    """Replace the core columns of ``existing`` with ``core``, keeping every other column."""
    if existing is None:
        return core
    extra = [c for c in existing.columns if c not in CORE_SCHEMA and c != KEY]
    return core.join(existing.select(KEY, *extra), on=KEY, how="left")


def describe(df: pl.DataFrame) -> str:
    on_disk = df.filter(pl.col("on_disk"))
    return json.dumps(
        {
            "episodes": df.height,
            "on_disk": on_disk.height,
            "public": on_disk.filter(pl.col("public")).height,
            "read_errors": on_disk.filter(pl.col("read_error").is_not_null()).height,
            "not_on_hub": on_disk.filter(pl.col("on_hub") == False).height,  # noqa: E712
        }
    )
