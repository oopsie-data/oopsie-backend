"""Select indexed episodes using recipe filters and deterministic sampling.

Excludes missing, unreadable, duplicate and critical episodes, derives outcome
labels, and filters by split, source, embodiment, length and outcome.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

import polars as pl
from oopsie_data_tools.annotation_tool.annotation_schema import OUTCOME_SUCCESS

from oopsie_backend.index.camera_names import UNIFIED_TYPE
from oopsie_backend.index.store import KEY

OPTIONAL = {
    "duplicate_of": pl.String,
    "critical": pl.Boolean,
    "eval_split": pl.String,
    "unified_cameras": UNIFIED_TYPE,
}
COLUMNS = [
    "dataset_repo", "repo_path", "repo_kind", "on_disk", "read_error", "episode_id", "lab_id", "robot_name",
    "gripper_name", "policy_name", "biarm", "binary_gripper", "dexterous_hand", "num_steps", "duration_seconds", "language_instruction", "task", "annotations",
]


def _annotation_time(timestamp: str | None) -> float:
    """Compare ISO and Unix timestamps, sorting undated annotations last."""
    if timestamp is None:
        return -math.inf
    try:
        seconds = float(timestamp)
    except ValueError:
        try:
            stamp = datetime.fromisoformat(timestamp.strip())
        except ValueError:
            return -math.inf
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    return seconds if math.isfinite(seconds) else -math.inf


def episode_labels(annotations: list[dict] | None) -> tuple[bool | None, list[str]]:
    latest_success, latest_time = None, -math.inf
    categories = set()
    for a in annotations or []:
        success = a["success"] if a["success"] is not None else OUTCOME_SUCCESS.get(a["outcome"] or "")
        if success is not None:
            timestamp = _annotation_time(a.get("timestamp"))
            if latest_success is None or timestamp > latest_time:
                latest_success, latest_time = float(success) >= 0.5, timestamp
        categories.update(a["failure_category"] or [])
    return latest_success, sorted(categories)


def select_episodes(index: pl.DataFrame, recipe: dict) -> pl.DataFrame:
    sel = recipe["selection"]
    conditions = [
        pl.col("on_disk"),
        pl.col("read_error").is_null(),
        pl.col("duplicate_of").is_null(),
        ~pl.col("critical").fill_null(False),
        pl.col("eval_split").fill_null("train").is_in(sel["splits"]),
        pl.col("repo_kind").is_in(sel["repo_kinds"]),
        pl.col("num_steps").fill_null(0) >= sel["min_steps"],
        pl.col("duration_seconds").fill_null(0) >= (sel["min_seconds"] or 0),
    ]
    for column, include, exclude in (
        ("dataset_repo", sel["datasets"], sel["exclude_datasets"]),
        ("lab_id", sel["labs"], sel["exclude_labs"]),
        ("robot_name", sel["robots"], None),
    ):
        if include:
            conditions.append(pl.col(column).is_in(include))
        if exclude:
            conditions.append(~pl.col(column).is_in(exclude).fill_null(False))
    for flag in ("biarm", "binary_gripper", "dexterous_hand"):
        if sel[flag] is not None:
            conditions.append(pl.col(flag).fill_null(False) == sel[flag])
    df = index.filter(pl.all_horizontal(conditions))
    labels = [episode_labels(a) for a in df["annotations"].to_list()]
    df = df.with_columns(
        pl.Series("success", [s for s, _ in labels], dtype=pl.Boolean),
        pl.Series("failure_categories", [c for _, c in labels], dtype=pl.List(pl.String)),
    )
    outcome = sel["outcome"]
    if outcome == "annotated":
        df = df.filter(pl.col("success").is_not_null())
    elif outcome == "success":
        df = df.filter(pl.col("success"))
    elif outcome == "failure":
        df = df.filter(~pl.col("success"))
    rank = pl.col(KEY).map_elements(
        lambda k: hashlib.sha1(f"{sel['seed']}:{k}".encode()).hexdigest(), return_dtype=pl.String
    )
    df = df.with_columns(rank.alias("_rank")).sort("_rank").drop("_rank")
    if sel["max_episodes"]:
        df = df.head(sel["max_episodes"])
    return df.sort("dataset_repo", "repo_path")
