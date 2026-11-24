"""Outcome, failure category and severity statistics of a labels frame (``common``).

The same functions serve the contributors' labels over the whole dataset and the
annotators' labels over the eval set. Category statistics use failed labellings only;
a labelling with several categories counts once for each, and one without any counts
as ``uncategorized``.
"""

from __future__ import annotations

from collections import Counter

import polars as pl

from oopsie_backend.index.store import KEY
from oopsie_backend.stats.analysis.common import (
    FAILURE,
    MISSING,
    OTHER,
    POLICY_CLASSES,
    SEVERITIES,
    TELEOP,
    UNCATEGORIZED,
    category_shares,
    shares,
)


def labelled(labels: pl.DataFrame, episodes: pl.DataFrame) -> pl.DataFrame:
    """Labels joined with the episode columns the statistics group by."""
    cols = [KEY, "policy_class", "policy_family", "frontier_group", "hardware", "duration_seconds"]
    return labels.join(episodes.select(cols), on=KEY, how="inner")


def failed(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("outcome") == FAILURE)


def _by(df: pl.DataFrame, column: str, min_n: int) -> dict:
    out = {}
    for (value,), group in df.group_by(column, maintain_order=False):
        out[str(value)] = category_shares(group["categories"].to_list(), min_n)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def failure_types(df: pl.DataFrame) -> dict:
    """Entry 15: outcomes, and categories of failed labellings."""
    f = failed(df)
    cats = [c for cats in f["categories"].to_list() for c in (cats or [UNCATEGORIZED])]
    return {
        "outcomes": shares(o or MISSING for o in df["outcome"].to_list()),
        "categories": shares(cats, total=f.height),
    }


def categories_per_episode(df: pl.DataFrame) -> dict:
    """Entry 18: number of distinct categories per failed labelling, and how often two categories are
    given together (``a + b``, as a share of failed labellings)."""
    cats = [sorted(set(c or [])) for c in failed(df)["categories"].to_list()]
    pairs = [f"{a} + {b}" for c in cats for i, a in enumerate(c) for b in c[i + 1:]]
    return {"categories": shares(len(c) for c in cats), "pairs": shares(pairs, total=len(cats))}


def by_policy(df: pl.DataFrame, min_n: int) -> dict:
    """Entry 21: category shares per policy class, per frontier group, and released against fine-tuned frontier families."""
    f = failed(df)
    frontier = f.filter(pl.col("policy_class") == "frontier")
    released_vs_tuned = {}
    families = f.filter(pl.col("policy_class") == "frontier")["policy_family"].unique().to_list()
    for family in sorted(families):
        fam = f.filter(pl.col("policy_family") == family)
        released_vs_tuned[family] = {
            cls: category_shares(fam.filter(pl.col("policy_class") == cls)["categories"].to_list(), min_n)
            for cls in ("frontier", "custom")
        }
    return {
        "policy_class": _by(f, "policy_class", min_n),
        "frontier_group": _by(frontier, "frontier_group", min_n),
        "released_vs_fine_tuned": released_vs_tuned,
    }


def _severities(df: pl.DataFrame) -> dict:
    return shares(s if s in SEVERITIES else MISSING for s in df["severity"].to_list())


def teleop_vs_policy(df: pl.DataFrame, min_n: int) -> dict:
    """Entry 22: categories and severities of teleoperation against policy episodes."""
    f = failed(df)
    groups = {
        "teleop": f.filter(pl.col("policy_class") == TELEOP),
        "policy": f.filter(pl.col("policy_class").is_in(POLICY_CLASSES)),
    }
    return {
        name: {"categories": category_shares(g["categories"].to_list(), min_n), "severity": _severities(g)}
        for name, g in groups.items()
    }


def severity(df: pl.DataFrame, min_n: int) -> dict:
    """Entry 23: severities of failed labellings overall, per policy class and per category."""
    f = failed(df)
    per_class = {
        str(cls): _severities(g) for (cls,), g in f.group_by("policy_class") if g.height >= min_n
    }
    exploded = f.with_columns(
        pl.when(pl.col("categories").list.len() > 0).then(pl.col("categories")).otherwise(pl.lit([UNCATEGORIZED]))
    ).explode("categories")
    per_category = {str(c): _severities(g) for (c,), g in exploded.group_by("categories") if g.height >= min_n}
    overall = _severities(f)
    return {
        "overall": overall,
        "catastrophic_share": overall["shares"].get("catastrophic", 0.0),
        "policy_class": per_class,
        "category": per_category,
    }


def by_hardware(df: pl.DataFrame, min_n: int) -> dict:
    """Entry 24: category shares per hardware group; hardware is confounded with policy and lab."""
    return _by(failed(df), "hardware", min_n)


def by_verb(df: pl.DataFrame, verbs: dict[str, list[str]], top: list[str], min_n: int) -> dict:
    """Entry 25: category shares per instruction verb; ``verbs`` maps episode keys to their verbs."""
    groups: dict[str, list[list[str]]] = {v: [] for v in top}
    for key, cats in failed(df).select(KEY, "categories").iter_rows():
        for verb in set(verbs.get(key, [])):
            if verb in groups:
                groups[verb].append(cats)
    return {verb: category_shares(cats, min_n) for verb, cats in groups.items()}


def taxonomy_coverage(df: pl.DataFrame) -> dict:
    """Entry 28: share of categorized failed labellings that include ``other``; uncategorized ones separately."""
    f = failed(df)
    cats = f["categories"].to_list()
    categorized = [c for c in cats if c]
    with_other = sum(OTHER in c for c in categorized)
    return {
        "failed": len(cats),
        "categorized": len(categorized),
        "other_share": with_other / len(categorized) if categorized else None,
        "uncategorized_share": (len(cats) - len(categorized)) / len(cats) if cats else None,
    }


def label_statistics(df: pl.DataFrame, verbs: dict[str, list[str]] | None, top_verbs: list[str], min_n: int) -> dict:
    """Every label-based entry, keyed by entry."""
    out = {
        "15_failure_types": failure_types(df),
        "18_categories_per_episode": categories_per_episode(df),
        "21_failure_types_by_policy": by_policy(df, min_n),
        "22_teleop_vs_policy": teleop_vs_policy(df, min_n),
        "23_severity": severity(df, min_n),
        "24_failure_types_by_hardware": by_hardware(df, min_n),
        "28_taxonomy_coverage": taxonomy_coverage(df),
    }
    if verbs is not None:
        out["25_failure_types_by_verb"] = by_verb(df, verbs, top_verbs, min_n)
    return out


def most_common(values, n: int) -> list[str]:
    return [v for v, _ in Counter(values).most_common(n)]
