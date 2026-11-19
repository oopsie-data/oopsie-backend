"""Statistics computed from the episode index alone (plans/stats.md, entries marked as index-based).

The failure statistics use the contributors' labels. The language entries (1, 2, 3, the
verbs of 5, and 25) need the ``analysis`` extra; ``language=False`` leaves them out.
"""

from __future__ import annotations

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
from oopsie_backend.stats.analysis.language import Parser, clusters, embed, normalize

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


def word_clouds(lang: Language) -> dict:
    """Entry 2: verbs and objects counted once per episode, and steps per instruction."""
    return {
        "verbs": shares(v for vs in lang.verbs.values() for v in set(vs)),
        "objects": shares(o for os_ in lang.objects.values() for o in os_),
        "steps_per_instruction": shares(lang.steps.values()),
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


def episode_length(episodes: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """Entry 6: duration in seconds overall, per policy class, per outcome and per failure category."""
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


def compute(episodes: pl.DataFrame, cfg: dict, with_language: bool = True) -> dict:
    """Every index-based entry, keyed by entry."""
    labels = contributor_labels(episodes)
    lang = language(episodes, cfg) if with_language else None
    min_n = cfg["min_group_episodes"]
    out = {}
    if lang:
        out["1_task_diversity"] = task_diversity(lang, cfg)
        out["2_word_clouds"] = word_clouds(lang)
        out["3_verb_object_coverage"] = verb_object_coverage(lang, cfg)
    out["5_hardware"] = hardware(episodes, lang, cfg)
    out["6_episode_length"] = episode_length(episodes, labels)
    out["7_cameras"] = cameras(episodes)
    out["9_policy_mix"] = policy_mix(episodes)
    out["10_control_setup"] = control_setup(episodes)
    out["12_data_growth"] = data_growth(episodes)
    top_verbs = failures.most_common((v for vs in lang.verbs.values() for v in set(vs)), cfg["top_verbs"]) if lang else []
    out.update(
        failures.label_statistics(failures.labelled(labels, episodes), lang.verbs if lang else None, top_verbs, min_n)
    )
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("_", 1)[0])))
