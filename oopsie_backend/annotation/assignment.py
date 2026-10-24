"""Assign eval episodes to annotators with replication and balanced labs, tasks and splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations

from oopsie_backend.splits.eval_split import EXCLUDED


def eval_samples(proposal: dict) -> list[dict]:
    """Every episode to annotate in a proposal, tagged with its split."""
    return [
        {**e, "split": split}
        for split, episodes in proposal["episodes"].items()
        if split != EXCLUDED
        for e in episodes
    ]


def assign(
    samples: list[dict],
    annotators: list[str],
    copies: dict[str, int],
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Give each sample to ``copies[split]`` distinct annotators.

    Each copy goes to the annotator with the fewest samples of that split,
    lab and task among those whose load is within one of the lowest load;
    ties prefer annotators who share the fewest samples with those already
    chosen, then a random order.
    """
    too_many = {s: c for s, c in copies.items() if c > len(annotators)}
    if too_many:
        raise ValueError(f"copies exceed {len(annotators)} annotators: {too_many}")
    rng = random.Random(seed)
    order = list(samples)
    rng.shuffle(order)

    load: Counter = Counter()
    counts = {f: defaultdict(Counter) for f in ("split", "lab_id", "task")}
    shared: Counter = Counter()
    out: dict[str, list[dict]] = {a: [] for a in annotators}
    for sample in order:
        chosen: list[str] = []
        for _ in range(copies[sample["split"]]):
            floor = min(load[a] for a in annotators if a not in chosen)
            candidates = [a for a in annotators if a not in chosen]
            pick = min(
                candidates,
                key=lambda a: (
                    max(0, load[a] - floor - 1),
                    counts["split"][a][sample["split"]],
                    counts["lab_id"][a][sample["lab_id"]],
                    counts["task"][a][(sample["lab_id"], sample["task"])],
                    sum(shared[frozenset((a, c))] for c in chosen),
                    rng.random(),
                ),
            )
            chosen.append(pick)
            load[pick] += 1
            counts["split"][pick][sample["split"]] += 1
            counts["lab_id"][pick][sample["lab_id"]] += 1
            counts["task"][pick][(sample["lab_id"], sample["task"])] += 1
        for a, b in combinations(chosen, 2):
            shared[frozenset((a, b))] += 1
        for a in chosen:
            out[a].append({**sample, "co_annotators": sorted(set(chosen) - {a})})
    return {a: spread_order(items, rng) for a, items in out.items()}


def spread_order(items: list[dict], rng: random.Random) -> list[dict]:
    """Random order that avoids the same lab twice in a row where possible."""
    remaining = list(items)
    rng.shuffle(remaining)
    ordered: list[dict] = []
    while remaining:
        prev = ordered[-1]["lab_id"] if ordered else None
        i = next((j for j, r in enumerate(remaining) if r["lab_id"] != prev), 0)
        ordered.append(remaining.pop(i))
    return ordered


def balance_report(assignment: dict[str, list[dict]]) -> dict:
    """Per-annotator loads and the spread of per-lab and per-split counts across annotators."""
    annotators = list(assignment)
    per = {
        f: {a: Counter(r[f] for r in items) for a, items in assignment.items()}
        for f in ("split", "lab_id")
    }

    def spread(field: str) -> dict[str, list[int]]:
        keys = sorted(set().union(*per[field].values()))
        return {k: [min(per[field][a][k] for a in annotators), max(per[field][a][k] for a in annotators)] for k in keys}

    pairs: Counter = Counter()
    for a, items in assignment.items():
        for r in items:
            for b in r["co_annotators"]:
                if a < b:
                    pairs[(a, b)] += 1
    pair_values = [pairs[(a, b)] for a, b in combinations(sorted(annotators), 2)]
    max_same_lab_run = {}
    for a, items in assignment.items():
        run = best = 0
        for i, r in enumerate(items):
            run = run + 1 if i and r["lab_id"] == items[i - 1]["lab_id"] else 1
            best = max(best, run)
        max_same_lab_run[a] = best
    return {
        "load": {a: len(items) for a, items in assignment.items()},
        "split_min_max": spread("split"),
        "lab_min_max": spread("lab_id"),
        "shared_pairs_min_max": [min(pair_values), max(pair_values)] if pair_values else [0, 0],
        "max_same_lab_run": max_same_lab_run,
    }
