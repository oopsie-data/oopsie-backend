"""New train bundles, so the annotated train set grows with every annotator.

A new train bundle holds ``size`` episodes, all in the current train pool: on disk, readable,
neither duplicate nor critical nor excluded by the annotation flags. Up to ``calibration`` of them
come from other annotators' train bundles, so that the new annotator shares episodes with
someone else and their agreement can be measured. These are the episodes with the fewest copies, preferring ones
whose annotation was returned, spread over as many source bundles as possible. The rest are
episodes of the chosen proposal's train pool that no bundle holds yet, spread over labs,
scenes, tasks, policies and success labels like the train sample (``splits.sample_train``),
with the budget going first to the strata earlier train bundles cover least.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import polars as pl

from oopsie_backend.annotation.assignment import spread_order
from oopsie_backend.annotation.bundles import sample_uid
from oopsie_backend.annotation.roster import TRAIN, Holding, Registry
from oopsie_backend.index import store
from oopsie_backend.splits.eval_split import (
    DEFAULT_ROBOT_ALIASES,
    TRAIN_ANNOTATED,
    apply_robot_aliases,
    coverage_sample,
    key,
    load_robot_aliases,
    train_pool,
)
from oopsie_backend.splits.planning import read_index
from oopsie_backend.splits.sample_train import LEVELS


class TrainPool:
    """The chosen proposal's train pool, without episodes the annotation flags exclude, and every
    index row, keyed by ``(dataset_repo, repo_path)``."""

    def __init__(self, index: Path, proposal: Path, robot_aliases: Path = DEFAULT_ROBOT_ALIASES):
        rows = read_index(index)
        apply_robot_aliases(rows, load_robot_aliases(robot_aliases))
        self.proposal = json.loads(proposal.read_text())
        flags = store.read_index(index, columns=[], optional={"annotation_excluded": pl.Boolean})
        excluded = set(flags.filter(pl.col("annotation_excluded").fill_null(False))[store.KEY].to_list())
        self.rows = [r for r in train_pool(rows, self.proposal) if r["episode_key"] not in excluded]
        self.keys = {key(r) for r in self.rows}
        self.by_key = {key(r): r for r in rows}


def calibration_entries(
    registry: Registry, pool: TrainPool, annotator: str, n: int, returned: set[str], rng: random.Random
) -> list[dict]:
    """Up to ``n`` train pool episodes of other annotators' train bundles."""
    held_by = registry.episodes(TRAIN)
    entries: dict[tuple[str, str], dict] = {}
    for bid in sorted(registry.bundles):
        if registry.bundles[bid]["kind"] == TRAIN:
            for e in registry.bundles[bid]["episodes"]:
                entries.setdefault(key(e), e)
    seen = registry.seen(annotator)
    candidates = [k for k in held_by if k not in seen and k in pool.keys]
    tiebreak = {k: rng.random() for k in candidates}
    used: Counter = Counter()
    picked = []
    for _ in range(min(n, len(candidates))):
        best = min(
            candidates,
            key=lambda k: (len(held_by[k]), sample_uid(*k) not in returned, used[held_by[k][0]], tiebreak[k]),
        )
        candidates.remove(best)
        picked.append(best)
        used[held_by[best][0]] += 1
    return [{**entries[k], "calibration": True} for k in picked]


def fresh_entries(registry: Registry, pool: TrainPool, n: int, rng: random.Random) -> list[dict]:
    """Up to ``n`` train pool episodes that no bundle holds."""
    taken = registry.episodes()
    available = [r for r in pool.rows if key(r) not in taken]
    existing = [pool.by_key[k] for k in registry.episodes(TRAIN) if k in pool.by_key]
    picked = coverage_sample(available, n, LEVELS, rng, existing)
    return [
        {
            "dataset_repo": r["dataset_repo"],
            "repo_path": r["repo_path"],
            "lab_id": r["lab_id"],
            "robot_name": r["robot_name"],
            "robot": r["robot"],
            "task": r["task"],
            "split": TRAIN_ANNOTATED,
        }
        for r in sorted(picked, key=key)
    ]


def add_train_bundle(
    registry: Registry,
    pool: TrainPool,
    annotator: str,
    size: int,
    calibration: int,
    returned: set[str],
    seed: int,
    note: str,
) -> Holding:
    """Build a train bundle for ``annotator`` and add it to the roster, under the roster lock."""
    with registry.writing():
        rng = random.Random(f"{seed}:{len(registry.bundles)}")
        cal = calibration_entries(registry, pool, annotator, min(calibration, size), returned, rng)
        fresh = fresh_entries(registry, pool, size - len(cal), rng)
        if not fresh:
            raise ValueError("the train pool has no episodes left that no bundle holds")
        entries = spread_order(cal + fresh, rng)
        return registry.add(
            TRAIN, entries, annotator, note,
            proposal=f"{pool.proposal['proposal_id']}_{TRAIN_ANNOTATED}", calibration=len(cal),
        )
