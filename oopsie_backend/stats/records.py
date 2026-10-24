"""Per-episode stats records from the episode index.

Only readable, on-disk episodes of contributor and external repos that are
neither a duplicate copy nor critical (``postqa.critical``) are used; the public
and internal filters in ``episode_stats`` then apply as before. Annotations come
from the index's decoded annotation schema: ``success`` falls back to the outcome when the file has no success
value, and ``failure_category`` is the list of normalized categories.
"""

from __future__ import annotations

import polars as pl
from oopsie_data_tools.annotation_tool.annotation_schema import OUTCOME_SUCCESS

from oopsie_backend.common.repos import CONTRIBUTOR, EXTERNAL
from oopsie_backend.stats.episode_stats import (
    UNKNOWN_CATEGORY,
    UNKNOWN_LAB,
    UNKNOWN_POLICY,
    UNKNOWN_ROBOT,
    UNKNOWN_SEVERITY,
)


def stats_episodes(df: pl.DataFrame) -> pl.DataFrame:
    """Episodes the stats count."""
    return df.filter(
        pl.col("on_disk")
        & pl.col("read_error").is_null()
        & pl.col("duplicate_of").is_null()
        & ~pl.col("critical").fill_null(False)
        & pl.col("repo_kind").is_in([CONTRIBUTOR, EXTERNAL])
    )


def _annotation(a: dict) -> dict:
    success = a["success"] if a["success"] is not None else OUTCOME_SUCCESS.get(a["outcome"] or "")
    description = (a["episode_description"] or "").strip()
    return {
        "name": a["name"],
        "timestamp": a["timestamp"],
        "success": success,
        "failure_category": a["failure_category"] or UNKNOWN_CATEGORY,
        "severity": a["severity"] or UNKNOWN_SEVERITY,
        "has_failure_description": bool(description),
        "failure_description": description,
    }


def records_from_index(df: pl.DataFrame) -> list[dict]:
    """Records in the shape ``episode_stats.aggregate`` and the internal report expect."""
    out = []
    for r in stats_episodes(df).iter_rows(named=True):
        ts = r["timestamp"]
        out.append(
            {
                "dataset_repo": r["dataset_repo"],
                "repo_path": r["repo_path"],
                "episode_id": r["episode_id"] or "",
                "lab_id": r["lab_id"] or UNKNOWN_LAB,
                "operator_name": r["operator_name"],
                "language_instruction": r["language_instruction"],
                "timestamp": ts.replace(microsecond=0).isoformat().replace("+00:00", "Z") if ts else None,
                "robot_name": r["robot_name"] or UNKNOWN_ROBOT,
                "policy_name": r["policy_name"] or UNKNOWN_POLICY,
                "action_space": r["action_space"] or [],
                "cameras_present": r["cameras"] or [],
                "duration_seconds": r["duration_seconds"],
                "annotations": [_annotation(a) for a in r["annotations"] or []],
            }
        )
    return out


def critical_summary(df: pl.DataFrame) -> dict:
    """Critical episodes per lab and in total, among readable, non-duplicate episodes on disk.

    ``qa_checked`` counts episodes with QA results; a QA check can only make a sampled
    episode critical, while read errors and duplicate videos apply to every episode.
    """
    rows = df.filter(
        pl.col("on_disk")
        & pl.col("duplicate_of").is_null()
        & pl.col("repo_kind").is_in([CONTRIBUTOR, EXTERNAL])
    )
    qa = pl.col("qa_checked").list.len().fill_null(0) > 0 if "qa_checked" in rows.columns else pl.lit(False)
    rows = rows.with_columns(
        pl.col("lab_id").fill_null(UNKNOWN_LAB).alias("lab"),
        qa.alias("_qa"),
        pl.col("critical").fill_null(False).alias("_critical"),
    )

    def summarize(part: pl.DataFrame) -> dict:
        reasons = (
            part.filter(pl.col("_critical")).select("critical_reasons").explode("critical_reasons")
            .group_by("critical_reasons").len().sort("len", descending=True).iter_rows()
        )
        episodes, critical = part.height, int(part["_critical"].sum())
        return {
            "episodes": episodes,
            "qa_checked": int(part["_qa"].sum()),
            "critical": critical,
            "critical_fraction": critical / episodes if episodes else 0.0,
            "reasons": {r: n for r, n in reasons if r},
        }

    return {
        "totals": summarize(rows),
        "per_lab": {lab: summarize(part) for (lab,), part in sorted(rows.group_by("lab"), key=lambda g: g[0])},
    }
