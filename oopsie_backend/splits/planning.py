"""Flat per-episode rows for split planning, read from the parquet index.

Only episodes that are on disk, public and readable, and neither a duplicate
copy nor critical, are returned. JSONL files written by earlier versions of the index builder are
still readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from oopsie_backend.index.core import normalize_task
from oopsie_backend.index.store import read_index as read_parquet_index

__all__ = ["normalize_task", "read_index", "planning_rows"]


def _primary(annotations: list[dict]) -> dict | None:
    """Latest annotation with a success label."""
    labelled = [a for a in annotations or [] if a["success"] is not None]
    if not labelled:
        return None
    return max(labelled, key=lambda a: (bool(a["timestamp"]), a["timestamp"] or ""))


def planning_rows(df: pl.DataFrame) -> list[dict]:
    df = df.filter(
        pl.col("on_disk")
        & pl.col("public")
        & pl.col("read_error").is_null()
        & pl.col("duplicate_of").is_null()
        & ~pl.col("critical").fill_null(False)
    )
    rows = []
    for r in df.iter_rows(named=True):
        primary = _primary(r["annotations"])
        rows.append(
            {
                "episode_key": r["episode_key"],
                "dataset_repo": r["dataset_repo"],
                "repo_path": r["repo_path"],
                "episode_id": r["episode_id"] or "",
                "lab_id": r["lab_id"],
                "robot_name": r["robot_name"],
                "policy_name": r["policy_name"],
                "operator_name": r["operator_name"] or "",
                "language_instruction": r["language_instruction"] or "",
                "task": r["task"] or "",
                "cameras": ",".join(r["cameras"] or []),
                "timestamp": r["timestamp"].isoformat().replace("+00:00", "Z") if r["timestamp"] else None,
                "duration_seconds": r["duration_seconds"],
                "success": primary["success"] if primary else None,
                "has_failure_description": bool(primary and (primary["episode_description"] or "").strip()),
                "annotators": sorted({a["name"] for a in r["annotations"] or []}),
            }
        )
    return rows


def read_index(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return planning_rows(read_parquet_index(path, optional={"duplicate_of": pl.String, "critical": pl.Boolean}))
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
