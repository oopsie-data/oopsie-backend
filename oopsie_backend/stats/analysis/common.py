"""Episode selection, labels and counting helpers shared by the analysis groups.

A labels frame has one row per labelling of an episode: ``episode_key``, ``source``
(``contributor`` or an annotator), ``outcome``, ``categories`` (list) and ``severity``.
The contributors' labels have one row per episode, the annotators' labels one row per
annotator and episode.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.common.repos import CONTRIBUTOR
from oopsie_backend.index.store import KEY

ANALYSIS_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "analysis.yaml"
FAILURE = "failure"
INCOMPLETE = "incomplete"
SUCCESS_OUTCOMES = ("success", "success_suboptimal", "success_side_effect")
OUTCOMES = (FAILURE, *SUCCESS_OUTCOMES, INCOMPLETE)
SEVERITIES = ("low", "medium", "catastrophic")
MISSING = "missing"
UNCATEGORIZED = "uncategorized"
OTHER = "other"
TELEOP = "teleop"
# Classes that count as policy episodes against teleoperation.
POLICY_CLASSES = ("frontier", "custom", "rl", "dagger", "scripted")
LABELS_SCHEMA = {
    KEY: pl.String,
    "source": pl.String,
    "outcome": pl.String,
    "categories": pl.List(pl.String),
    "severity": pl.String,
}
EPISODE_COLUMNS = [
    "dataset_repo", "repo_path", "robot_name", "gripper_name", "repo_kind", "on_disk", "read_error", "duplicate_of", "critical", "lab_id",
    "language_instruction", "duration_seconds", "control_freq", "action_space", "unified_cameras",
    "hub_uploaded_at", "annotations", "policy_family", "policy_class", "frontier_group",
    "unified_arm", "unified_end_effector", "end_effector_type", "embodiment", "biarm", "uses_mobile_base",
]

# Written by the annotation flags stage; null until it runs.
FLAG_COLUMNS = {"annotation_excluded": pl.Boolean, "annotation_unusable": pl.List(pl.String)}


def load_config(path: Path = ANALYSIS_CONFIG) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def contributed(index: pl.DataFrame) -> pl.DataFrame:
    """Readable, on-disk contributed episodes that are neither a duplicate copy, critical nor excluded by the annotation flags."""
    df = index.filter(
        (pl.col("repo_kind") == CONTRIBUTOR)
        & pl.col("on_disk")
        & pl.col("read_error").is_null()
        & pl.col("duplicate_of").is_null()
        & ~pl.col("critical").fill_null(False)
        & ~pl.col("annotation_excluded").fill_null(False)
    )
    layout = (
        pl.when(pl.col("uses_mobile_base").fill_null(False)).then(pl.lit("mobile_")).otherwise(pl.lit(""))
        + pl.when(pl.col("biarm").fill_null(False)).then(pl.lit("bimanual")).otherwise(pl.lit("single_arm"))
    )
    return df.with_columns(
        layout=layout,
        hardware=layout + pl.lit("/") + pl.col("end_effector_type").fill_null(MISSING),
    )


def _primary(annotations: list[dict] | None) -> dict | None:
    """The latest annotation that records a timestamp, else any; the same choice as the public stats."""
    if not annotations:
        return None
    return max(annotations, key=lambda a: (1 if a.get("timestamp") else 0, a.get("timestamp") or ""))


def contributor_labels(episodes: pl.DataFrame) -> pl.DataFrame:
    """Labels frame of the contributors' primary annotation of every episode."""
    rows = []
    for key, records in episodes.select(KEY, "annotations").iter_rows():
        a = _primary(records)
        if a is None:
            continue
        rows.append({
            KEY: key,
            "source": "contributor",
            "outcome": a.get("outcome") or None,
            "categories": sorted(set(a.get("failure_category") or [])),
            "severity": a.get("severity") or None,
        })
    return pl.DataFrame(rows, schema=LABELS_SCHEMA)


def counts(values: Iterable) -> dict[str, int]:
    """Occurrences of every value, most frequent first."""
    return dict(Counter(str(v) for v in values).most_common())


def shares(values: Iterable, total: int | None = None) -> dict:
    """``{n, counts, shares}``; ``total`` is the denominator when values overcount episodes."""
    c = counts(values)
    n = total if total is not None else sum(c.values())
    return {"n": n, "counts": c, "shares": {k: v / n for k, v in c.items()} if n else {}}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a share ``k / n``."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def category_shares(categories: list[list[str]], min_n: int = 0) -> dict:
    """Share of labellings that contain each category, with Wilson intervals; ``None`` below ``min_n``."""
    n = len(categories)
    if n < min_n or n == 0:
        return {"n": n, "below_min": True}
    c = Counter(cat for cats in categories for cat in (cats or [UNCATEGORIZED]))
    return {
        "n": n,
        "shares": {k: {"count": v, "share": v / n, "ci": wilson(v, n)} for k, v in c.most_common()},
    }


def distribution(values: Iterable[float]) -> dict:
    """Count, mean and quantiles of numeric values."""
    s = pl.Series([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=pl.Float64)
    if s.is_empty():
        return {"n": 0}
    return {
        "n": s.len(),
        "mean": s.mean(),
        **{f"q{int(q * 100):02d}": s.quantile(q, "linear") for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }


def effective_number(values: Iterable) -> float:
    """``exp(entropy)`` of the distribution of ``values``."""
    c = Counter(values)
    n = sum(c.values())
    return math.exp(-sum(v / n * math.log(v / n) for v in c.values())) if n else 0.0


def anonymize(names: Iterable[str]) -> dict[str, str]:
    """Stable anonymous ids (``A1``, ``A2``, …) ordered by a hash of each name."""
    ordered = sorted(set(names), key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    return {name: f"A{i + 1}" for i, name in enumerate(ordered)}
