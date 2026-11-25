"""Statistics of the annotators' returns with failure segments (plans/stats.md, eval set entries).

The returns are read from the index's ``annotation_*`` columns (``annotation.columns``)
into three frames:

- outcomes: one row per annotator and episode, with ``episode_key``, ``annotator``,
  ``outcome``, ``severity``, ``instruction`` (the annotator's language instruction
  where it differs from the episode's) and ``description``;
- segments: one row per segment and category, with ``episode_key``, ``annotator``,
  ``segment`` (the segment's number within the annotator's episode), ``start`` and
  ``end`` (seconds from the episode start), ``type``, ``category`` and ``label``;
- assigned: one row per assigned episode, with ``episode_key``, ``split`` and
  ``assigned`` (the number of annotators it was given to).

Each annotator's latest record counts. Records made on a copy whose ``content_hash``
differs from the episode's current one are left out.

A segment with several categories is paired once; its category label joins them with ``+``.

An annotator's categories of an episode are the union of its segment categories.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations

import polars as pl

from oopsie_backend.annotation.columns import current_records
from oopsie_backend.index.store import KEY
from oopsie_backend.stats.analysis import failures
from oopsie_backend.stats.analysis.agreement import (
    intersect_intervals,
    jaccard,
    krippendorff_alpha,
    merge_intervals,
    pair_segments,
    union_iou,
)
from oopsie_backend.stats.analysis.common import (
    FAILURE,
    INCOMPLETE,
    LABELS_SCHEMA,
    MISSING,
    SEVERITIES,
    SUCCESS_OUTCOMES,
    UNCATEGORIZED,
    anonymize,
    category_shares,
    contributor_labels,
    distribution,
    shares,
    wilson,
)
from oopsie_backend.stats.analysis.language import content_words

OUTCOMES_SCHEMA = {
    KEY: pl.String,
    "annotator": pl.String,
    "outcome": pl.String,
    "severity": pl.String,
    "instruction": pl.String,
    "description": pl.String,
}
SEGMENTS_SCHEMA = {
    KEY: pl.String,
    "annotator": pl.String,
    "segment": pl.Int64,
    "start": pl.Float64,
    "end": pl.Float64,
    "type": pl.String,
    "category": pl.String,
    "label": pl.String,
}
ASSIGNED_SCHEMA = {KEY: pl.String, "split": pl.String, "assigned": pl.Int64}
# Episodes listed per end of the per-episode agreement ranking (entry 43).
RANKED_EPISODES = 10


RETURNS_COLUMNS = ["content_hash", "control_freq", "annotation_batch", "annotation_split", "annotation_assigned", "annotation_records"]


def returns_from_index(index: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(outcomes, segments, assigned)`` of the assigned episodes of ``index`` (``RETURNS_COLUMNS``
    and ``common.FLAG_COLUMNS``), without the episodes and records the annotation flags leave out."""
    outcomes, segments, assigned = [], [], []
    stale = unusable = 0
    rows = (
        index.filter(pl.col("annotation_batch").is_not_null() & ~pl.col("annotation_excluded").fill_null(False))
        .select(KEY, *RETURNS_COLUMNS, "annotation_unusable")
    )
    for row in rows.iter_rows(named=True):
        key, freq = row[KEY], row["control_freq"]
        assigned.append({KEY: key, "split": row["annotation_split"], "assigned": len(row["annotation_assigned"] or [])})
        records, n_stale = current_records(row["annotation_records"], row["content_hash"])
        stale += n_stale
        dropped = set(row["annotation_unusable"] or [])
        unusable += sum(r["annotator"] in dropped for r in records)
        for r in records:
            if r["annotator"] in dropped:
                continue
            annotator = r["annotator"]
            outcomes.append({
                KEY: key,
                "annotator": annotator,
                "outcome": r["outcome"],
                "severity": r["severity"] or None,
                "instruction": r["language_instruction"],
                "description": r["episode_description"] or None,
            })
            if not freq:
                continue
            for i, seg in enumerate(r["temporal_segments"] or []):
                base = {
                    KEY: key,
                    "annotator": annotator,
                    "segment": i,
                    "start": seg["start"] / freq,
                    "end": (seg["end"] + 1) / freq,
                    "type": seg["type"],
                    "label": seg["label"] or None,
                }
                for cat in seg["failure_category"] or [None]:
                    segments.append(base | {"category": cat})
    if stale:
        print(f"left out {stale} annotation(s) made on an earlier version of their episode", file=sys.stderr)
    if unusable:
        print(f"left out {unusable} annotation(s) the annotation flags mark unusable", file=sys.stderr)
    return (
        pl.DataFrame(outcomes, schema=OUTCOMES_SCHEMA),
        pl.DataFrame(segments, schema=SEGMENTS_SCHEMA),
        pl.DataFrame(assigned, schema=ASSIGNED_SCHEMA),
    )


def annotator_labels(outcomes: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    """Labels frame with one row per annotator and episode."""
    cats = segments.group_by(KEY, "annotator").agg(pl.col("category").drop_nulls().unique().sort().alias("categories"))
    df = outcomes.join(cats, on=[KEY, "annotator"], how="left").with_columns(
        pl.col("categories").fill_null(pl.lit([], dtype=pl.List(pl.String))),
    )
    return df.select(KEY, pl.col("annotator").alias("source"), "outcome", "categories", "severity").cast(LABELS_SCHEMA)


TIMING_BINS = 20
START_WITHIN = (0.5, 1, 2, 5, 10)  # entry 30: seconds between the starts of paired segments


def _segment_lists(segments: pl.DataFrame) -> dict[tuple[str, str], list[tuple[float, float, str | None]]]:
    grouped = segments.group_by(KEY, "annotator", "segment").agg(
        pl.col("start").first(), pl.col("end").first(), pl.col("category").drop_nulls().unique().sort()
    )
    out = defaultdict(list)
    for key, annotator, _, start, end, cats in grouped.sort("start").iter_rows():
        out[(key, annotator)].append((start, end, "+".join(cats) or None))
    return out


def time_until_failure(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Entry 14: start of the first segment over the episode duration, overall (with a histogram of
    ``TIMING_BINS`` equal bins over 0–1) and per policy class."""
    first = segments.group_by(KEY, "annotator").agg(pl.col("start").min())
    df = labels.join(first, left_on=[KEY, "source"], right_on=[KEY, "annotator"], how="inner").with_columns(
        (pl.col("start") / pl.col("duration_seconds")).clip(0, 1).alias("relative")
    )
    bins = (df["relative"] * TIMING_BINS).floor().clip(0, TIMING_BINS - 1).cast(pl.Int64).value_counts()
    histogram = dict(zip(*bins.sort("relative").to_dict(as_series=False).values()))
    return {
        "overall": distribution(df["relative"]),
        "histogram": [histogram.get(i, 0) for i in range(TIMING_BINS)],
        "policy_class": {str(k): distribution(g["relative"]) for (k,), g in df.group_by("policy_class")},
    }


def _segment_counts(labels: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    n = segments.group_by(KEY, "annotator").agg(pl.col("segment").n_unique().alias("segments"))
    return labels.join(n, left_on=[KEY, "source"], right_on=[KEY, "annotator"], how="left").with_columns(
        pl.col("segments").fill_null(0)
    )


def segments_per_episode(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Entry 17: number of segments per label, and per episode as the lower median over its annotators."""
    counts = _segment_counts(labels, segments)
    per_episode = counts.group_by(KEY).agg(pl.col("segments"))["segments"].to_list()
    return {
        "labels": shares(counts["segments"].to_list()),
        "episodes": shares(statistics.median_low(c) for c in per_episode),
    }


def recovery(labels: pl.DataFrame, segments: pl.DataFrame, min_n: int) -> dict:
    """Entry 19: share of episodes with segments whose outcome is a success."""
    df = _segment_counts(labels, segments).filter(pl.col("segments") > 0).with_columns(
        pl.col("outcome").is_in(SUCCESS_OUTCOMES).alias("recovered")
    )

    def share(g: pl.DataFrame) -> dict:
        k = int(g["recovered"].sum())
        return {"n": g.height, "recovered": k, "share": k / g.height if g.height else None, "ci": wilson(k, g.height)}

    per_category = {}
    exploded = df.explode("categories")
    for (cat,), g in exploded.group_by("categories"):
        if g.height >= min_n:
            per_category[str(cat or UNCATEGORIZED)] = share(g)
    return {
        "overall": share(df),
        "policy_class": {str(k): share(g) for (k,), g in df.group_by("policy_class") if g.height >= min_n},
        "category": per_category,
    }


def contributor_agreement(annotators: pl.DataFrame, contributors: pl.DataFrame) -> dict:
    """Entry 27: contributors' against annotators' labels, per field, and the category confusion."""
    df = annotators.join(contributors, on=KEY, how="inner", suffix="_contributor")
    outcome_units = [[a, c] for a, c in df.select("outcome", "outcome_contributor").iter_rows() if a and c]
    severity_units = [
        [a, c] for a, c in df.select("severity", "severity_contributor").iter_rows() if a in SEVERITIES and c in SEVERITIES
    ]
    failed = df.filter((pl.col("outcome") == FAILURE) & (pl.col("outcome_contributor") == FAILURE))
    every = sorted({c for cs in failed["categories"].to_list() + failed["categories_contributor"].to_list() for c in cs})
    per_category = {
        cat: krippendorff_alpha([[cat in a, cat in c] for a, c in failed.select("categories", "categories_contributor").iter_rows()])
        for cat in every
    }
    confusion = Counter()
    for a, c in failed.select("categories", "categories_contributor").iter_rows():
        for ca in a or [UNCATEGORIZED]:
            for cc in c or [UNCATEGORIZED]:
                confusion[(cc, ca)] += 1
    return {
        "episodes": df[KEY].n_unique(),
        "outcome_alpha": krippendorff_alpha(outcome_units),
        "outcome_agreement": sum(a == c for a, c in outcome_units) / len(outcome_units) if outcome_units else None,
        "failure": _contributor_failure(df),
        "severity_alpha": krippendorff_alpha(severity_units, order=SEVERITIES),
        "category_alpha": per_category,
        "confusion_contributor_to_annotator": {f"{c} -> {a}": n for (c, a), n in confusion.most_common()},
    }


def _contributor_failure(df: pl.DataFrame) -> dict:
    """Failure or not: contributor against every annotator label, the annotators' majority and unanimous labels."""
    df = df.filter(pl.col("outcome").is_not_null() & pl.col("outcome_contributor").is_not_null())
    singles = [(a == FAILURE, c == FAILURE) for a, c in df.select("outcome", "outcome_contributor").iter_rows()]
    majority, unanimous, units = [], [], []
    disagreements = Counter()
    for _, g in df.group_by(KEY):
        votes = [o == FAILURE for o in g["outcome"]]
        contributor = g["outcome_contributor"][0] == FAILURE
        k = sum(votes)
        if 2 * k == len(votes):
            continue
        maj = 2 * k > len(votes)
        units.append([contributor, maj])
        majority.append(maj == contributor)
        if k in (0, len(votes)):
            unanimous.append(maj == contributor)
        if maj != contributor:
            disagreements["contributor_failure" if contributor else "contributor_not_failure"] += 1
    share = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "labels": len(singles),
        "label_agreement": share([a == c for a, c in singles]),
        "alpha": krippendorff_alpha([[a, c] for a, c in singles]),
        "majority_episodes": len(majority),
        "majority_agreement": share(majority),
        "majority_alpha": krippendorff_alpha(units),
        "unanimous_episodes": len(unanimous),
        "unanimous_agreement": share(unanimous),
        "majority_disagreements": dict(disagreements),
    }


def _pairs(segments: pl.DataFrame) -> list[dict]:
    """Paired segments of every two annotators of an episode, with unpaired counts."""
    lists = _segment_lists(segments)
    by_episode = defaultdict(dict)
    for (key, annotator), segs in lists.items():
        by_episode[key][annotator] = segs
    out = []
    for key, annotators in by_episode.items():
        for x, y in combinations(sorted(annotators), 2):
            a, b = annotators[x], annotators[y]
            pairs = pair_segments([(s, e) for s, e, _ in a], [(s, e) for s, e, _ in b])
            out.append({
                KEY: key,
                "counts": (len(a), len(b)),
                "pairs": [(a[i], b[j], overlap) for i, j, overlap in pairs],
            })
    return out


def annotator_agreement(labels: pl.DataFrame, pairs: list[dict]) -> dict:
    """Entry 29: α over the outcome, every category per episode and the category of paired segments."""
    by_episode = defaultdict(list)
    for row in labels.iter_rows(named=True):
        by_episode[row[KEY]].append(row)
    multi = {k: rows for k, rows in by_episode.items() if len(rows) > 1}
    every = sorted({c for rows in multi.values() for r in rows for c in r["categories"]})
    return {
        "episodes": len(multi),
        "outcome_alpha": krippendorff_alpha([[r["outcome"] for r in rows if r["outcome"]] for rows in multi.values()]),
        "category_alpha": {
            cat: krippendorff_alpha([[cat in r["categories"] for r in rows] for rows in multi.values()]) for cat in every
        },
        "segment_category_alpha": krippendorff_alpha([[a[2], b[2]] for p in pairs for a, b, _ in p["pairs"]]),
        "failure": _failure_agreement(multi),
        "category_sets": _category_sets(multi),
    }


def _failure_agreement(multi: dict[str, list[dict]]) -> dict:
    """Failure or not between annotators: α and pair agreement, overall and by labels per episode."""
    by_n = defaultdict(list)
    for rows in multi.values():
        votes = [r["outcome"] == FAILURE for r in rows if r["outcome"]]
        by_n[len(votes)].extend(a == b for a, b in combinations(votes, 2))
    every = [x for xs in by_n.values() for x in xs]
    return {
        "alpha": krippendorff_alpha([[r["outcome"] == FAILURE for r in rows if r["outcome"]] for rows in multi.values()]),
        "pairs": len(every),
        "pair_agreement": sum(every) / len(every) if every else None,
        "pair_agreement_by_labels": {str(n): sum(xs) / len(xs) for n, xs in sorted(by_n.items()) if xs},
    }


def _category_sets(multi: dict[str, list[dict]]) -> dict:
    """Category sets of annotators who both call an episode a failure, per pair and per episode."""
    pairs = []
    shared_any, shared_all = [], []
    for rows in multi.values():
        failed = [set(r["categories"]) for r in rows if r["outcome"] == FAILURE]
        pairs += list(combinations(failed, 2))
        if len(failed) > 1:
            shared_any.append(any(a & b for a, b in combinations(failed, 2)))
        if len(failed) > 2:
            shared_all.append(bool(set.intersection(*failed)))
    n = len(pairs)
    return {
        "pairs": n,
        "identical": sum(a == b for a, b in pairs) / n if n else None,
        "share_a_category": sum(bool(a & b) for a, b in pairs) / n if n else None,
        "mean_jaccard": sum(jaccard(a, b) for a, b in pairs) / n if n else None,
        "episodes_two_share": {"n": len(shared_any), "share": sum(shared_any) / len(shared_any) if shared_any else None},
        "episodes_three_share": {"n": len(shared_all), "share": sum(shared_all) / len(shared_all) if shared_all else None},
    }


def _covered(segments: pl.DataFrame) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """episode -> annotator -> merged time its segments cover."""
    spans = segments.unique([KEY, "annotator", "segment"])
    raw = defaultdict(lambda: defaultdict(list))
    for key, annotator, start, end in spans.select(KEY, "annotator", "start", "end").iter_rows():
        raw[key][annotator].append((start, end))
    return {k: {a: merge_intervals(iv) for a, iv in v.items()} for k, v in raw.items()}


def _union_agreement(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """IoU of the time every two annotators cover, with pairs where only one marks segments counted as 0."""
    covered = _covered(segments)
    kinds = Counter()
    ious = []
    for key, rows in labels.group_by(KEY):
        annotators = sorted(rows["source"].to_list())
        for x, y in combinations(annotators, 2):
            a, b = covered.get(key[0], {}).get(x, []), covered.get(key[0], {}).get(y, [])
            kind = "both" if a and b else "one" if a or b else "none"
            kinds[kind] += 1
            if kind != "none":
                ious.append(union_iou(a, b))
    return {"pairs": dict(kinds), "iou": distribution(ious)}


def _three_annotators(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Episodes with three annotators: how many mark segments, and whether all three cover a common moment."""
    three = {k[0]: g["source"].to_list() for k, g in labels.group_by(KEY) if g.height == 3}

    def count(segs: pl.DataFrame) -> dict:
        covered = _covered(segs)
        marking = Counter()
        pairwise = common = 0
        for key, annotators in three.items():
            u = [covered.get(key, {}).get(a, []) for a in annotators]
            m = sum(bool(x) for x in u)
            marking[str(m)] += 1
            if m == 3:
                pairwise += all(intersect_intervals(a, b) for a, b in combinations(u, 2))
                common += bool(intersect_intervals(intersect_intervals(u[0], u[1]), u[2]))
        n3 = marking["3"]
        return {
            "annotators_marking": dict(sorted(marking.items())),
            "every_pair_overlaps": pairwise,
            "common_moment": common,
            "common_moment_share": common / n3 if n3 else None,
        }

    return {
        "episodes": len(three),
        "all_segments": count(segments),
        "failure_segments": count(segments.filter(pl.col("type") == FAILURE)),
    }


def temporal_agreement(pairs: list[dict], labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Entry 30: overlap (with a histogram of ``TIMING_BINS`` equal bins over 0–1) and start difference of paired
    segments (with the share within each of ``START_WITHIN`` seconds), and agreement on the number of segments.

    Also the IoU of the time two annotators cover, which counts pairs where only one marks
    segments, and the common moment of three annotators.
    """
    overlaps = [o for p in pairs for _, _, o in p["pairs"]]
    starts = [abs(a[0] - b[0]) for p in pairs for a, b, _ in p["pairs"]]
    total = sum(sum(p["counts"]) for p in pairs)
    unpaired = sum(sum(p["counts"]) - 2 * len(p["pairs"]) for p in pairs)
    return {
        "annotator_pairs": len(pairs),
        "iou": distribution(overlaps),
        "iou_histogram": [
            sum(min(int(o * TIMING_BINS), TIMING_BINS - 1) == i for o in overlaps) for i in range(TIMING_BINS)
        ],
        "start_difference_seconds": distribution(starts),
        "start_within_seconds": {str(t): sum(d <= t for d in starts) / len(starts) if starts else None for t in START_WITHIN},
        "same_segment_count_share": sum(p["counts"][0] == p["counts"][1] for p in pairs) / len(pairs) if pairs else None,
        "unpaired_segment_share": unpaired / total if total else None,
        "covered_time": _union_agreement(labels, segments),
        "three_annotators": _three_annotators(labels, segments),
    }


def severity_agreement(labels: pl.DataFrame) -> dict:
    """Entry 31: ordinal α over the episode severity."""
    units = defaultdict(list)
    for key, sev in labels.select(KEY, "severity").iter_rows():
        if sev in SEVERITIES:
            units[key].append(sev)
    return {"alpha": krippendorff_alpha(list(units.values()), order=SEVERITIES)}


def category_confusion(pairs: list[dict]) -> dict:
    """Entry 32: categories of paired segments, counted in both directions."""
    c = Counter()
    for p in pairs:
        for a, b, _ in p["pairs"]:
            c[(a[2] or UNCATEGORIZED, b[2] or UNCATEGORIZED)] += 1
            c[(b[2] or UNCATEGORIZED, a[2] or UNCATEGORIZED)] += 1
    return {f"{x} / {y}": n for (x, y), n in c.most_common()}


def annotator_bias(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Entry 33: every anonymized annotator's label distributions against the pool."""
    df = _segment_counts(labels, segments)
    ids = anonymize(df["source"].unique().to_list())

    def summary(g: pl.DataFrame) -> dict:
        f = g.filter(pl.col("outcome") == FAILURE)
        return {
            "episodes": g.height,
            "outcomes": shares(o or MISSING for o in g["outcome"].to_list())["shares"],
            "severity": shares(s if s in SEVERITIES else MISSING for s in f["severity"].to_list())["shares"],
            "categories": category_shares(f["categories"].to_list()).get("shares", {}),
            "segments_per_episode": distribution(g["segments"].cast(pl.Float64)),
        }

    return {
        "pool": summary(df),
        "annotators": {ids[str(a)]: summary(g) for (a,), g in df.group_by("source")},
    }


def failures_without_segment(labels: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """Entry 37: share of failed episodes without a segment per policy class; incomplete episodes counted apart."""
    df = _segment_counts(labels, segments)
    f = df.filter(pl.col("outcome") == FAILURE)

    def share(g: pl.DataFrame) -> dict:
        k = int((g["segments"] == 0).sum())
        return {"failed": g.height, "without_segment": k, "share": k / g.height if g.height else None}

    return {
        "overall": share(f),
        "policy_class": {str(k): share(g) for (k,), g in f.group_by("policy_class")},
        "incomplete": int((df["outcome"] == INCOMPLETE).sum()),
    }


def annotation_coverage(outcomes: pl.DataFrame, assigned: pl.DataFrame) -> dict:
    """Entry 42: assigned episodes per split by the number of annotator labels returned."""
    n = outcomes.group_by(KEY).agg(pl.col("annotator").n_unique().alias("labels"))
    df = assigned.join(n, on=KEY, how="left").with_columns(pl.col("labels").fill_null(0))

    def summary(g: pl.DataFrame) -> dict:
        top = max(int(g["assigned"].max() or 0), int(g["labels"].max() or 0))
        c = Counter(g["labels"].to_list())
        return {
            "episodes": g.height,
            "assigned": sorted(set(g["assigned"].to_list())),
            "labels": {str(k): {"n": c[k], "share": c[k] / g.height} for k in range(top + 1)},
            "complete_share": float((g["labels"] >= g["assigned"]).mean()),
        }

    return {
        "overall": summary(df) if df.height else {"episodes": 0},
        "split": {str(k): summary(g) for (k,), g in sorted(df.group_by("split"), key=lambda kv: str(kv[0][0]))},
        "returned_labels": outcomes.height,
        "assigned_labels": int(assigned["assigned"].sum()),
    }


def _outcome_score(a: str | None, b: str | None) -> float:
    """1 for the same outcome, 0.5 when both or neither are a failure, else 0."""
    if a == b:
        return 1.0
    return 0.5 if (a == FAILURE) == (b == FAILURE) else 0.0


def episode_scores(labels: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    """One row per episode with several labels: mean over annotator pairs of the outcome, category and covered-time agreement.

    ``score`` is the mean of the three parts.
    """
    covered = _covered(segments)
    n_segments = segments.group_by(KEY, "annotator").agg(pl.col("segment").n_unique())
    n_seg = {(k, a): n for k, a, n in n_segments.iter_rows()}
    rows = []
    for (key,), g in labels.group_by(KEY):
        if g.height < 2:
            continue
        anns = g.sort("source").to_dicts()
        cov = covered.get(key, {})
        parts = [
            (
                _outcome_score(x["outcome"], y["outcome"]),
                jaccard(set(x["categories"]), set(y["categories"])),
                union_iou(cov.get(x["source"], []), cov.get(y["source"], [])),
            )
            for x, y in combinations(anns, 2)
        ]
        o, c, t = (sum(p[i] for p in parts) / len(parts) for i in range(3))
        failed = sum(r["outcome"] == FAILURE for r in anns)
        rows.append({
            KEY: key,
            "labels": g.height,
            "score": (o + c + t) / 3,
            "outcome": o,
            "category": c,
            "covered_time": t,
            "failure_votes": failed,
            "marked": any(cov.values()),
            "summary": [
                f"{r['outcome']}/{'+'.join(r['categories']) or '-'}/{n_seg.get((key, r['source']), 0)} segments" for r in anns
            ],
        })
    schema = {
        KEY: pl.String, "labels": pl.Int64, "score": pl.Float64, "outcome": pl.Float64, "category": pl.Float64,
        "covered_time": pl.Float64, "failure_votes": pl.Int64, "marked": pl.Boolean, "summary": pl.List(pl.String),
    }
    return pl.DataFrame(rows, schema=schema)


def _ranked(df: pl.DataFrame, episodes: pl.DataFrame, descending: bool) -> list[dict]:
    top = df.sort(["score", KEY], descending=[descending, False]).head(RANKED_EPISODES)
    top = top.join(episodes.select(KEY, "lab_id", "language_instruction"), on=KEY, how="left", maintain_order="left")
    cols = [KEY, "lab_id", "language_instruction", "score", "outcome", "category", "covered_time", "summary"]
    return top.select(cols).to_dicts()


def episode_agreement(scores: pl.DataFrame, episodes: pl.DataFrame) -> dict:
    """Entry 43: distribution of the per-episode agreement score, and the least and most agreed episodes.

    The most agreed episodes are taken among those that every annotator calls a failure
    and that have segments, since labellings without categories or segments agree trivially.
    """
    unanimous_failure = scores.filter((pl.col("failure_votes") == pl.col("labels")) & pl.col("marked"))
    return {
        "episodes": scores.height,
        "score": distribution(scores["score"]),
        "by_labels": {str(k): distribution(g["score"]) for (k,), g in sorted(scores.group_by("labels"))},
        "with_failure_label": distribution(scores.filter(pl.col("failure_votes") > 0)["score"]),
        "perfect": int((scores["score"] == 1).sum()),
        "perfect_without_segments": int(((scores["score"] == 1) & ~scores["marked"]).sum()),
        "least_agreed": _ranked(scores, episodes, descending=False),
        "most_agreed_failures": _ranked(unanimous_failure, episodes, descending=True),
    }


def agreement_by_verb(scores: pl.DataFrame, verbs: dict[str, list[str]], top_verbs: list[str]) -> dict:
    """Entry 44: per-episode agreement per instruction verb, over all episodes and over episodes with a failure label."""
    rows = [
        {KEY: k, "verb": v} for k in scores[KEY].to_list() for v in set(verbs.get(k, [])) if v in top_verbs
    ]
    if not rows:
        return {}
    df = pl.DataFrame(rows).join(scores, on=KEY)
    split = (pl.col("failure_votes") > 0) & (pl.col("failure_votes") < pl.col("labels"))

    def summary(g: pl.DataFrame) -> dict:
        return {
            "episodes": g.height,
            "score": g["score"].mean(),
            "outcome": g["outcome"].mean(),
            "category": g["category"].mean(),
            "covered_time": g["covered_time"].mean(),
            "split_vote_share": float(g.select(split.mean()).item()),
            "any_failure_share": float((g["failure_votes"] > 0).mean()),
        }

    with_failure = df.filter(pl.col("failure_votes") > 0)
    return {
        "all": {v: summary(g) for (v,), g in sorted(df.group_by("verb"), key=lambda kv: kv[1]["score"].mean())},
        "with_failure_label": {
            v: summary(g) for (v,), g in sorted(with_failure.group_by("verb"), key=lambda kv: kv[1]["score"].mean())
        },
    }


def changed_instructions(outcomes: pl.DataFrame, episodes: pl.DataFrame, contributors: pl.DataFrame) -> dict:
    """Entry 45: episodes where an annotator changed the language instruction, with every annotator's outcome.

    Annotators are anonymized as in entry 33.
    """
    keys = outcomes.filter(pl.col("instruction").is_not_null())[KEY].unique().to_list()
    df = outcomes.filter(pl.col(KEY).is_in(keys))
    ids = anonymize(outcomes["annotator"].unique().to_list())
    op = dict(contributors.select(KEY, "outcome").iter_rows())
    info = {r[KEY]: r for r in episodes.filter(pl.col(KEY).is_in(keys)).select(KEY, "lab_id", "language_instruction").iter_rows(named=True)}
    listed, agree = [], {"changed": [], "kept": []}
    direction = Counter()
    disagreeing = 0
    for (key,), g in sorted(df.group_by(KEY), key=lambda kv: (str(info.get(kv[0][0], {}).get("lab_id")), kv[0][0])):
        contributor = op.get(key)
        votes = [o == FAILURE for o in g["outcome"]]
        split = 0 < sum(votes) < len(votes)
        disagreeing += split
        for o, instruction in g.select("outcome", "instruction").iter_rows():
            if contributor is None or o is None:
                continue
            same = (o == FAILURE) == (contributor == FAILURE)
            agree["changed" if instruction else "kept"].append(same)
            if instruction and not same:
                direction["contributor_failure" if contributor == FAILURE else "contributor_not_failure"] += 1
        listed.append({
            KEY: key,
            "lab_id": info.get(key, {}).get("lab_id"),
            "instruction": info.get(key, {}).get("language_instruction"),
            "contributor_outcome": contributor,
            "annotators_disagree": split,
            "labels": [
                {"annotator": ids[a], "outcome": o, "instruction": i}
                for a, o, i in g.sort("annotator").select("annotator", "outcome", "instruction").iter_rows()
            ],
        })
    share = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "episodes": len(keys),
        "changed_labels": int(outcomes["instruction"].is_not_null().sum()),
        "annotators_disagree_on_failure": disagreeing,
        "contributor_agreement": {"changed": share(agree["changed"]), "kept": share(agree["kept"])},
        "changed_disagreements": dict(direction),
        "episodes_listed": listed,
    }


def description_words(outcomes: pl.DataFrame, segments: pl.DataFrame, cfg: dict) -> dict:
    """Entry 47: content words of the annotators' failure descriptions, i.e. the descriptions of labels with a
    failure outcome and the labels of all segments, each counted once per text (``language.content_words``);
    ``verbs`` keeps only verbs and adjectives, which name what went wrong rather than the objects."""
    texts = outcomes.filter(pl.col("outcome") == FAILURE)["description"].drop_nulls().to_list()
    texts += segments.unique([KEY, "annotator", "segment"])["label"].drop_nulls().to_list()
    words = content_words(texts, cfg)
    verbs = content_words(texts, cfg, pos={"VERB", "ADJ"})
    return {"texts": len(texts), "counts": dict(words.most_common()), "verbs": dict(verbs.most_common())}


def compute(
    episodes: pl.DataFrame,
    outcomes: pl.DataFrame,
    segments: pl.DataFrame,
    cfg: dict,
    verbs: dict[str, list[str]] | None = None,
    assigned: pl.DataFrame | None = None,
) -> dict:
    """Every annotation-based entry, keyed by entry. ``episodes`` comes from ``common.contributed``."""
    keys = episodes[KEY].to_list()
    outcomes = outcomes.filter(pl.col(KEY).is_in(keys))
    segments = segments.filter(pl.col(KEY).is_in(keys))
    min_n = cfg["min_group_episodes"]
    labels = failures.labelled(annotator_labels(outcomes, segments), episodes)
    pairs = _pairs(segments)
    top_verbs = failures.most_common((v for vs in (verbs or {}).values() for v in set(vs)), cfg["top_verbs"])
    out = failures.label_statistics(labels, verbs, top_verbs, min_n)
    contributors = contributor_labels(episodes.filter(pl.col(KEY).is_in(outcomes[KEY].unique().to_list())))
    scores = episode_scores(labels, segments)
    out.update({
        "14_time_until_failure": time_until_failure(labels, segments),
        "17_segments_per_episode": segments_per_episode(labels, segments),
        "19_recovery": recovery(labels, segments, min_n),
        "27_contributor_agreement": contributor_agreement(labels, contributors),
        "29_annotator_agreement": annotator_agreement(labels, pairs),
        "30_temporal_agreement": temporal_agreement(pairs, labels, segments),
        "31_severity_agreement": severity_agreement(labels),
        "32_category_confusion": category_confusion(pairs),
        "33_annotator_bias": annotator_bias(labels, segments),
        "37_failures_without_segment": failures_without_segment(labels, segments),
        "43_episode_agreement": episode_agreement(scores, episodes),
        "45_changed_instructions": changed_instructions(outcomes, episodes, contributors),
    })
    if assigned is not None and not assigned.is_empty():
        out["42_annotation_coverage"] = annotation_coverage(outcomes, assigned.filter(pl.col(KEY).is_in(keys)))
    if verbs:
        out["44_agreement_by_verb"] = agreement_by_verb(scores, verbs, top_verbs)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("_", 1)[0])))
