"""Select the episodes of a recipe from the episode index.

Every recipe leaves out episodes that are not on disk, unreadable, duplicate
copies or critical. ``selection.splits`` picks the training episodes (``train``,
those the eval split does not hold out) and any ``eval_split`` values. ``selection``
then filters by repo kind, dataset, lab,
robot, the embodiment flags of the index (``biarm``, ``binary_gripper``,
``dexterous_hand``), length in steps and seconds, and outcome, and ``max_episodes`` keeps a stable
pseudo-random subset ranked by a hash of the episode key and ``seed``.

An episode's ``success`` is the mean success of its annotations (falling back
to the annotation outcome); ``failure_categories`` is the union of their categories.
"""

from __future__ import annotations

import hashlib

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


def episode_labels(annotations: list[dict] | None) -> tuple[float | None, list[str]]:
    successes, categories = [], set()
    for a in annotations or []:
        success = a["success"] if a["success"] is not None else OUTCOME_SUCCESS.get(a["outcome"] or "")
        if success is not None:
            successes.append(float(success))
        categories.update(a["failure_category"] or [])
    return (sum(successes) / len(successes) if successes else None), sorted(categories)


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
        pl.Series("success", [s for s, _ in labels], dtype=pl.Float64),
        pl.Series("failure_categories", [c for _, c in labels], dtype=pl.List(pl.String)),
    )
    outcome = sel["outcome"]
    if outcome == "annotated":
        df = df.filter(pl.col("success").is_not_null())
    elif outcome == "success":
        df = df.filter(pl.col("success") >= 0.5)
    elif outcome == "failure":
        df = df.filter(pl.col("success") < 0.5)
    rank = pl.col(KEY).map_elements(
        lambda k: hashlib.sha1(f"{sel['seed']}:{k}".encode()).hexdigest(), return_dtype=pl.String
    )
    df = df.with_columns(rank.alias("_rank")).sort("_rank").drop("_rank")
    if sel["max_episodes"]:
        df = df.head(sel["max_episodes"])
    return df.sort("dataset_repo", "repo_path")
