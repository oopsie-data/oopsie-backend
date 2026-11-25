"""Statistics computed from the episode index alone (plans/stats.md, entries marked as index-based).

The failure statistics use the contributors' labels. The language entries (1, 2, 3, the
verbs of 5, and 25) need the ``analysis`` extra; ``language=False`` leaves them out.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import polars as pl

from oopsie_backend.index.store import KEY
from oopsie_backend.stats.analysis import failures
from oopsie_backend.stats.analysis.common import (
    FAILURE,
    MISSING,
    UNCATEGORIZED,
    contributor_labels,
    distribution,
    effective_number,
    shares,
)
from oopsie_backend.stats.analysis.language import Parser, clusters, content_words, embed, normalize

SUFFIX = re.compile(r"(_[2-9])?(_stereo)?$")
WRIST_VIEWS = {"wrist", "wrist_left", "wrist_right"}


@dataclass
class Language:
    """Per-episode verbs, objects and step counts, and the task cluster of every episode per threshold."""

    verbs: dict[str, list[str]]
    objects: dict[str, list[str]]
    steps: dict[str, int]
    clusters: dict[float, dict[str, int]]


def language(episodes: pl.DataFrame, cfg: dict) -> Language:
    keys = episodes[KEY].to_list()
    texts = [normalize(t) for t in episodes["language_instruction"].to_list()]
    parsed = Parser(cfg).parse_all(texts)
    verbs, objects, steps = {}, {}, {}
    for key, text in zip(keys, texts):
        parts = parsed.get(text, [])
        verbs[key] = [s.verb for s in parts if s.verb]
        objects[key] = sorted({o for s in parts for o in s.objects})
        steps[key] = len(parts)
    unique = sorted({t for t in texts if t})
    vectors = embed(unique, cfg["embedding_model"])
    thresholds = sorted(set(cfg.get("similarity_thresholds") or []) | {cfg["similarity_threshold"]})
    per_threshold = {}
    for threshold in thresholds:
        ids = dict(zip(unique, clusters(vectors, threshold).tolist()))
        per_threshold[threshold] = {k: ids[t] for k, t in zip(keys, texts) if t}
    return Language(verbs, objects, steps, per_threshold)


def task_diversity(lang: Language, cfg: dict) -> dict:
    """Entry 1: number of task clusters and effective number of tasks, per similarity threshold."""
    out = {
        str(t): {"clusters": len(set(ids.values())), "effective_tasks": effective_number(ids.values())}
        for t, ids in lang.clusters.items()
    }
    return {"threshold": cfg["similarity_threshold"], "per_threshold": out}


def word_clouds(lang: Language, episodes: pl.DataFrame, cfg: dict) -> dict:
    """Entry 2: verbs and objects counted once per episode, steps per instruction, and the content words of the
    instructions' first sentences, each counted once per episode (``language.content_words``)."""
    words = content_words(episodes["language_instruction"].to_list(), cfg, first_sentence=True)
    return {
        "verbs": shares(v for vs in lang.verbs.values() for v in set(vs)),
        "objects": shares(o for os_ in lang.objects.values() for o in os_),
        "steps_per_instruction": shares(lang.steps.values()),
        "words": {"n": episodes.height, "counts": dict(words.most_common())},
    }


def verb_object_coverage(lang: Language, cfg: dict) -> dict:
    """Entry 3: distinct verbs, objects and pairs, and the top-verbs × top-objects matrix with its empty share."""
    pairs = Counter()
    for key, vs in lang.verbs.items():
        for verb in set(vs):
            for obj in lang.objects.get(key, []):
                pairs[(verb, obj)] += 1
    verbs = Counter(v for vs in lang.verbs.values() for v in set(vs))
    objects = Counter(o for os_ in lang.objects.values() for o in os_)
    top_v = [v for v, _ in verbs.most_common(cfg["top_verbs"])]
    top_o = [o for o, _ in objects.most_common(cfg["top_objects"])]
    matrix = {v: {o: pairs.get((v, o), 0) for o in top_o} for v in top_v}
    cells = [n for row in matrix.values() for n in row.values()]
    return {
        "verbs": len(verbs),
        "objects": len(objects),
        "pairs": len(pairs),
        "pair_coverage": len(pairs) / (len(verbs) * len(objects)) if verbs and objects else None,
        "top_matrix": matrix,
        "top_matrix_empty_share": sum(n == 0 for n in cells) / len(cells) if cells else None,
    }


def hardware(episodes: pl.DataFrame, lang: Language | None, cfg: dict) -> dict:
    """Entry 5: per hardware group, episodes, task clusters, labs and verbs; overall layout and end effector shares."""
    groups = {}
    ids = lang.clusters[cfg["similarity_threshold"]] if lang else {}
    for (group,), g in episodes.group_by("hardware"):
        keys = g[KEY].to_list()
        entry = {"episodes": g.height, "labs": g["lab_id"].n_unique()}
        if lang:
            entry["task_clusters"] = len({ids[k] for k in keys if k in ids})
            entry["verbs"] = shares((v for k in keys for v in set(lang.verbs.get(k, []))), total=g.height)
        groups[str(group)] = entry
    n = episodes.height
    return {
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["episodes"])),
        "bimanual_share": episodes["biarm"].fill_null(False).sum() / n if n else None,
        "mobile_share": episodes["uses_mobile_base"].fill_null(False).sum() / n if n else None,
        "dexterous_hand_share": (episodes["end_effector_type"] == "dexterous_hand").sum() / n if n else None,
    }


LENGTH_BINS = 40
LENGTH_RANGE = (1.0, 1000.0)


def _log_histogram(values: pl.Series) -> list[int]:
    """Counts in ``LENGTH_BINS`` log-spaced bins over ``LENGTH_RANGE``; values outside fall into the end bins."""
    lo, hi = math.log10(LENGTH_RANGE[0]), math.log10(LENGTH_RANGE[1])
    idx = ((values.drop_nulls().clip(*LENGTH_RANGE).log10() - lo) / (hi - lo) * LENGTH_BINS).floor()
    counts = Counter(min(int(i), LENGTH_BINS - 1) for i in idx.to_list())
    return [counts.get(i, 0) for i in range(LENGTH_BINS)]


def episode_length(episodes: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """Entry 6: duration in seconds overall, per policy class, per outcome (with a log-binned histogram) and per
    failure category."""
    df = episodes.select(KEY, "duration_seconds", "policy_class").join(
        labels.select(KEY, "outcome", "categories"), on=KEY, how="left"
    )
    per_category: dict[str, list[float]] = {}
    for d, cats in df.filter(pl.col("outcome") == FAILURE).select("duration_seconds", "categories").iter_rows():
        for c in cats or [UNCATEGORIZED]:
            per_category.setdefault(c, []).append(d)
    return {
        "overall": distribution(df["duration_seconds"]),
        "policy_class": {str(k): distribution(g["duration_seconds"]) for (k,), g in df.group_by("policy_class")},
        "outcome": {str(k or MISSING): distribution(g["duration_seconds"]) for (k,), g in df.group_by("outcome")},
        "failure_category": {k: distribution(v) for k, v in per_category.items()},
        "outcome_histogram": {
            "edges": [10 ** (math.log10(LENGTH_RANGE[0]) + i * math.log10(LENGTH_RANGE[1] / LENGTH_RANGE[0]) / LENGTH_BINS)
                      for i in range(LENGTH_BINS + 1)],
            "counts": {str(k or MISSING): _log_histogram(g["duration_seconds"]) for (k,), g in df.group_by("outcome")},
        },
    }


def _views(cameras: list[dict] | None) -> list[str]:
    return [SUFFIX.sub("", c["name"]) for c in cameras or [] if c.get("name")]


def cameras(episodes: pl.DataFrame) -> dict:
    """Entry 7: cameras per episode, view frequency, wrist with exterior share, per hardware group."""

    def summary(g: pl.DataFrame) -> dict:
        views = [_views(c) for c in g["unified_cameras"].to_list()]
        both = sum(bool(WRIST_VIEWS & set(v)) and any(x.startswith("exterior") for x in v) for v in views)
        return {
            "cameras_per_episode": shares(len(v) for v in views),
            "views": shares((x for v in views for x in set(v)), total=len(views)),
            "wrist_and_exterior_share": both / len(views) if views else None,
        }

    return {
        "overall": summary(episodes),
        "hardware": {str(k): summary(g) for (k,), g in episodes.group_by("hardware")},
    }


def policy_mix(episodes: pl.DataFrame) -> dict:
    """Entry 9: episodes per policy class, frontier episodes per group, custom episodes per family."""
    return {
        "policy_class": shares(c or MISSING for c in episodes["policy_class"].to_list()),
        "frontier_group": shares(episodes.filter(pl.col("policy_class") == "frontier")["frontier_group"].to_list()),
        "custom_family": shares(episodes.filter(pl.col("policy_class") == "custom")["policy_family"].to_list()),
    }


def control_setup(episodes: pl.DataFrame) -> dict:
    """Entry 10: action spaces (an episode counts once per space) and control frequency."""
    spaces = [s for ss in episodes["action_space"].to_list() for s in set(ss or [])]
    return {
        "action_space": shares(spaces, total=episodes.height),
        "control_freq": distribution(episodes["control_freq"]),
        "control_freq_by_class": {
            str(k): distribution(g["control_freq"]) for (k,), g in episodes.group_by("policy_class")
        },
    }


def data_growth(episodes: pl.DataFrame) -> dict:
    """Entry 12: cumulative episodes per day of reaching the Hub."""
    dated = episodes.filter(pl.col("hub_uploaded_at").is_not_null())
    if dated.is_empty():
        return {"available": False, "reason": "hub_uploaded_at is empty; run the hub_upload_times stage"}
    daily = (
        dated.group_by(pl.col("hub_uploaded_at").dt.date().alias("day")).len().sort("day")
        .with_columns(pl.col("len").cum_sum().alias("cumulative"))
    )
    return {
        "available": True,
        "undated": episodes.height - dated.height,
        "days": {str(d): c for d, c in daily.select("day", "cumulative").iter_rows()},
    }


def composition_hours(episodes: pl.DataFrame, labels: pl.DataFrame, lang: Language | None) -> dict:
    """Entry 46: episodes and hours per verb, arm, policy class and contributor outcome.

    In ``verbs`` an episode counts once for each distinct verb of its instruction, so verb hours
    add up to more than the total. ``verb_sets`` counts every episode once, under its distinct
    verbs joined by ``+`` (empty without a verb), for grouping verbs without counting twice."""
    hours = dict(zip(episodes[KEY].to_list(), (episodes["duration_seconds"].fill_null(0) / 3600).to_list()))

    def per(values: dict[str, list[str]]) -> dict:
        out: dict[str, dict] = {}
        for key, vs in values.items():
            for v in set(vs):
                row = out.setdefault(v, {"episodes": 0, "hours": 0.0})
                row["episodes"] += 1
                row["hours"] += hours[key]
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["hours"]))

    def column(name: str) -> dict[str, list[str]]:
        return {k: [v or MISSING] for k, v in episodes.select(KEY, name).iter_rows()}

    outcome = dict(labels.select(KEY, "outcome").iter_rows())
    out = {
        "total_hours": sum(hours.values()),
        "arm": per(column("unified_arm")),
        "end_effector_type": per(column("end_effector_type")),
        "policy_class": per(column("policy_class")),
        "outcome": per({k: [outcome.get(k) or MISSING] for k in hours}),
    }
    if lang:
        out["verbs"] = per(lang.verbs)
        out["verb_sets"] = per({k: ["+".join(sorted(set(vs)))] for k, vs in lang.verbs.items()})
    return out


def compute(episodes: pl.DataFrame, cfg: dict, with_language: bool = True) -> dict:
    """Every index-based entry, keyed by entry."""
    labels = contributor_labels(episodes)
    lang = language(episodes, cfg) if with_language else None
    min_n = cfg["min_group_episodes"]
    out = {}
    if lang:
        out["1_task_diversity"] = task_diversity(lang, cfg)
        out["2_word_clouds"] = word_clouds(lang, episodes, cfg)
        out["3_verb_object_coverage"] = verb_object_coverage(lang, cfg)
    out["5_hardware"] = hardware(episodes, lang, cfg)
    out["6_episode_length"] = episode_length(episodes, labels)
    out["7_cameras"] = cameras(episodes)
    out["9_policy_mix"] = policy_mix(episodes)
    out["10_control_setup"] = control_setup(episodes)
    out["12_data_growth"] = data_growth(episodes)
    out["46_composition_hours"] = composition_hours(episodes, labels, lang)
    top_verbs = failures.most_common((v for vs in lang.verbs.values() for v in set(vs)), cfg["top_verbs"]) if lang else []
    out.update(
        failures.label_statistics(failures.labelled(labels, episodes), lang.verbs if lang else None, top_verbs, min_n)
    )
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("_", 1)[0])))
