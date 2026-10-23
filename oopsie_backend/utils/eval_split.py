"""Propose eval splits (ID, OD-task/ID-scene, OD-task/OD-scene) from an episode index.

A scene is approximated by ``(lab_id, robot_name, cameras)``; a task by the
normalized language instruction. Every proposal is checked against the
remaining training pool so violations show up in the report instead of being
silently accepted.
"""

from __future__ import annotations

import difflib
import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

NEAR_DUPLICATE_SIMILARITY = 0.85
EXCLUDED = "excluded"
ID_SPLIT = "id"
OD_TASK_SPLIT = "od_task_id_scene"
OOD_LAB_SPLIT = "od_task_od_scene"
SPLITS = (ID_SPLIT, OD_TASK_SPLIT, OOD_LAB_SPLIT)


@dataclass
class SplitConfig:
    id_frac: float = 0.05
    id_alpha: float = 0.5
    id_max_group_frac: float = 0.25
    id_min_group: int = 8
    ood_lab_min: int = 40
    ood_lab_max_frac: float = 0.03
    od_task_lab_min_frac: float = 0.02
    od_task_lab_max_frac: float = 0.15
    od_task_min_tasks: int = 3
    od_task_min_episodes: int = 25
    od_task_max_share: float = 0.4
    holdout_cap: int | None = None
    candidates_per_split: int = 3
    common_robots: list[str] = field(default_factory=list)
    custom_robots: list[str] = field(default_factory=list)
    seed: int = 0


def key(row: dict) -> tuple[str, str]:
    return row["dataset_repo"], row["repo_path"]


def scene(row: dict) -> tuple[str, str, str]:
    return row["lab_id"], row["robot_name"], row["cameras"]


def _hours(rows: list[dict]) -> float:
    return sum(r["duration_seconds"] or 0.0 for r in rows) / 3600


def _success_rate(rows: list[dict]) -> float | None:
    labeled = [r["success"] for r in rows if r["success"] is not None]
    return sum(s >= 0.5 for s in labeled) / len(labeled) if labeled else None


def _by(rows: list[dict], field_name: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r[field_name]].append(r)
    return out


def robot_classes(rows: list[dict], cfg: SplitConfig) -> dict[str, str]:
    """``common`` for robots used by at least two labs, ``custom`` otherwise; config overrides win."""
    labs_per_robot: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        labs_per_robot[r["robot_name"]].add(r["lab_id"])
    classes = {
        robot: "common" if len(labs) >= 2 else "custom"
        for robot, labs in labs_per_robot.items()
    }
    classes.update({r: "common" for r in cfg.common_robots})
    classes.update({r: "custom" for r in cfg.custom_robots})
    return classes


def lab_summary(rows: list[dict], cfg: SplitConfig) -> list[dict]:
    classes = robot_classes(rows, cfg)
    total = len(rows)
    out = []
    for lab, rs in sorted(_by(rows, "lab_id").items(), key=lambda kv: -len(kv[1])):
        robots = sorted({r["robot_name"] for r in rs})
        out.append(
            {
                "lab_id": lab,
                "episodes": len(rs),
                "frac": len(rs) / total,
                "hours": _hours(rs),
                "robots": robots,
                "embodiment": "custom"
                if any(classes[r] == "custom" for r in robots)
                else "common",
                "scenes": len({scene(r) for r in rs}),
                "tasks": len({r["task"] for r in rs}),
                "success_rate": _success_rate(rs),
            }
        )
    return out


def ood_lab_candidates(rows: list[dict], cfg: SplitConfig) -> dict[str, list[dict]]:
    """Small labs eligible for full holdout, smallest first, split by embodiment class."""
    total = len(rows)
    tasks_by_lab = {lab: {r["task"] for r in rs} for lab, rs in _by(rows, "lab_id").items()}
    out: dict[str, list[dict]] = {"common": [], "custom": []}
    for lab in sorted(lab_summary(rows, cfg), key=lambda s: s["episodes"]):
        if lab["episodes"] < cfg.ood_lab_min or lab["frac"] > cfg.ood_lab_max_frac:
            continue
        other_tasks = set().union(
            *(t for other, t in tasks_by_lab.items() if other != lab["lab_id"])
        )
        lab["shared_tasks"] = sorted(tasks_by_lab[lab["lab_id"]] & other_tasks)
        out[lab["embodiment"]].append(lab)
    return out


def task_similarity(a: str, b: str) -> float:
    """Max of character-level ``difflib`` ratio and word-set Jaccard, so word swaps count as similar."""
    wa, wb = set(a.split()), set(b.split())
    jaccard = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(), jaccard)


def nearest_task(task: str, pool: set[str]) -> tuple[str, float]:
    """Most similar task string in ``pool`` by ``task_similarity``."""
    best, best_sim = "", 0.0
    for other in pool:
        sim = task_similarity(task, other)
        if sim > best_sim:
            best, best_sim = other, sim
    return best, best_sim


def od_task_candidates(
    rows: list[dict], cfg: SplitConfig, excluded_labs: set[str]
) -> list[dict]:
    """Tasks of medium multi-task labs that can be held out with their scenes kept in training.

    A task qualifies when it is absent from every other lab, holds between
    ``od_task_min_episodes`` episodes and ``od_task_max_share`` of its lab, and
    each of its scenes still has training episodes on other tasks.
    """
    total = len(rows)
    all_tasks = {r["task"] for r in rows}
    labs_per_task: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        labs_per_task[r["task"]].add(r["lab_id"])
    out = []
    for lab, rs in _by(rows, "lab_id").items():
        frac = len(rs) / total
        if lab in excluded_labs or not (
            cfg.od_task_lab_min_frac <= frac <= cfg.od_task_lab_max_frac
        ):
            continue
        tasks = Counter(r["task"] for r in rs)
        big_tasks = [t for t, n in tasks.items() if n >= cfg.od_task_min_episodes]
        if len(big_tasks) < cfg.od_task_min_tasks:
            continue
        for task in big_tasks:
            n = tasks[task]
            share = n / len(rs)
            if share > cfg.od_task_max_share or labs_per_task[task] != {lab}:
                continue
            held = [r for r in rs if r["task"] == task]
            kept_scenes = {scene(r) for r in rs if r["task"] != task}
            if not {scene(r) for r in held} <= kept_scenes:
                continue
            similar, similarity = nearest_task(task, all_tasks - {task})
            out.append(
                {
                    "lab_id": lab,
                    "task": task,
                    "nearest_task": similar,
                    "similarity": similarity,
                    "episodes": n,
                    "lab_share": share,
                    "lab_tasks": len(tasks),
                    "success_rate": _success_rate(held),
                }
            )
    # Prefer tasks near a quarter of their lab: large enough to evaluate, small enough to keep the lab.
    out.sort(key=lambda c: (abs(c["lab_share"] - 0.25), -c["episodes"]))
    return out


def _allocate(sizes: dict, n_target: int, alpha: float, cap_frac: float) -> dict:
    """Split ``n_target`` across groups ∝ size**alpha, capped at ``cap_frac`` of each group."""
    caps = {g: math.floor(s * cap_frac) for g, s in sizes.items()}
    alloc = {g: 0 for g in sizes}
    open_groups = {g for g, c in caps.items() if c > 0}
    remaining = n_target
    while remaining > 0 and open_groups:
        weights = {g: sizes[g] ** alpha for g in open_groups}
        norm = sum(weights.values())
        quota = {g: remaining * w / norm for g, w in weights.items()}
        step = {g: min(caps[g] - alloc[g], math.floor(q)) for g, q in quota.items()}
        if not any(step.values()):
            for g in sorted(open_groups, key=lambda g: -(quota[g] % 1))[:remaining]:
                step[g] = 1
        for g, s in step.items():
            alloc[g] += s
            remaining -= s
        open_groups = {g for g in open_groups if alloc[g] < caps[g]}
    return alloc


def _subsample(rs: list[dict], n: int, alpha: float, rng: random.Random) -> list[dict]:
    """Keep ``n`` of ``rs``, stratified over ``(robot, cameras, task)``."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rs:
        groups[(*scene(r), r["task"])].append(r)
    alloc = _allocate({g: len(v) for g, v in groups.items()}, n, alpha, 1.0)
    kept = []
    for g in sorted(alloc):
        kept.extend(rng.sample(sorted(groups[g], key=lambda r: r["repo_path"]), alloc[g]))
    return kept


def sample_id(
    pool: list[dict], n_target: int, cfg: SplitConfig, rng: random.Random
) -> tuple[list[dict], dict]:
    """Stratified sample over ``(lab, robot, cameras, task)`` groups.

    Groups smaller than ``id_min_group`` are skipped since removing any of
    their episodes would leave the task nearly unseen in training.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        groups[(*scene(r), r["task"])].append(r)
    eligible = {g: rs for g, rs in groups.items() if len(rs) >= cfg.id_min_group}
    alloc = _allocate(
        {g: len(rs) for g, rs in eligible.items()},
        n_target,
        cfg.id_alpha,
        cfg.id_max_group_frac,
    )
    picked = []
    for g in sorted(alloc):
        if alloc[g]:
            rs = sorted(eligible[g], key=lambda r: r["repo_path"])
            picked.extend(rng.sample(rs, alloc[g]))
    info = {
        "groups": len(groups),
        "eligible_groups": len(eligible),
        "sampled_groups": sum(1 for v in alloc.values() if v),
        "ineligible_episodes": sum(len(rs) for g, rs in groups.items() if g not in eligible),
    }
    return picked, info


def check_proposal(train: list[dict], splits: dict[str, list[dict]]) -> list[str]:
    """Human-readable violations of the split definitions."""
    problems = []
    train_scenes = {scene(r) for r in train}
    train_tasks = {r["task"] for r in train}
    train_groups = {(*scene(r), r["task"]) for r in train}
    train_labs = {r["lab_id"] for r in train}
    train_robots = {r["robot_name"] for r in train}

    missing = [r for r in splits[ID_SPLIT] if (*scene(r), r["task"]) not in train_groups]
    if missing:
        problems.append(f"{ID_SPLIT}: {len(missing)} episode(s) whose scene+task is absent from train")
    for r in splits[OD_TASK_SPLIT]:
        if r["task"] in train_tasks:
            problems.append(f"{OD_TASK_SPLIT}: task '{r['task']}' also in train")
            break
    lost_scenes = {scene(r) for r in splits[OD_TASK_SPLIT]} - train_scenes
    if lost_scenes:
        problems.append(f"{OD_TASK_SPLIT}: scene(s) absent from train: {sorted(lost_scenes)}")
    leaked = {r["lab_id"] for r in splits[OOD_LAB_SPLIT]} & train_labs
    if leaked:
        problems.append(f"{OOD_LAB_SPLIT}: lab(s) also in train: {sorted(leaked)}")
    gone = {r["robot_name"] for r in splits[OOD_LAB_SPLIT]} - train_robots
    if gone:
        problems.append(f"note: robot(s) only in {OOD_LAB_SPLIT}, unseen in train: {sorted(gone)}")
    shared = {r["task"] for r in splits[OOD_LAB_SPLIT]} & train_tasks
    if shared:
        problems.append(f"note: {OOD_LAB_SPLIT} shares {len(shared)} task(s) with train: {sorted(shared)[:5]}")
    return problems


def split_stats(rows: list[dict], total: int) -> dict:
    return {
        "episodes": len(rows),
        "frac": len(rows) / total if total else 0.0,
        "hours": _hours(rows),
        "labs": len({r["lab_id"] for r in rows}),
        "robots": len({r["robot_name"] for r in rows}),
        "tasks": len({r["task"] for r in rows}),
        "success_rate": _success_rate(rows),
        "with_failure_description": sum(r["has_failure_description"] for r in rows),
        "multi_annotator": sum(len(r["annotators"]) >= 2 for r in rows),
    }


def make_proposal(
    rows: list[dict],
    cfg: SplitConfig,
    ood_labs: list[str],
    od_tasks: list[tuple[str, str]],
    seed: int,
) -> dict:
    """Assign held-out labs, held-out ``(lab, task)`` pairs and a stratified ID sample."""
    ood_set = set(ood_labs)
    od_set = set(od_tasks)
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    pool = []
    for r in rows:
        if r["lab_id"] in ood_set:
            splits[OOD_LAB_SPLIT].append(r)
        elif (r["lab_id"], r["task"]) in od_set:
            splits[OD_TASK_SPLIT].append(r)
        else:
            pool.append(r)
    rng = random.Random(seed)
    excluded: list[dict] = []
    if cfg.holdout_cap is not None:
        for name, field_fn in ((OOD_LAB_SPLIT, lambda r: r["lab_id"]), (OD_TASK_SPLIT, lambda r: (r["lab_id"], r["task"]))):
            capped = []
            for unit in sorted({field_fn(r) for r in splits[name]}):
                rs = [r for r in splits[name] if field_fn(r) == unit]
                if len(rs) > cfg.holdout_cap:
                    kept = _subsample(rs, cfg.holdout_cap, cfg.id_alpha, rng)
                    kept_keys = {key(r) for r in kept}
                    excluded.extend(r for r in rs if key(r) not in kept_keys)
                    rs = kept
                capped.extend(rs)
            splits[name] = capped
    n_id = round(cfg.id_frac * len(rows))
    splits[ID_SPLIT], id_info = sample_id(pool, n_id, cfg, rng)
    id_keys = {key(r) for r in splits[ID_SPLIT]}
    train = [r for r in pool if key(r) not in id_keys]

    train_tasks = {r["task"] for r in train}
    nearest = {
        f"{lab}::{task}": nearest_task(task, train_tasks) for lab, task in od_tasks
    }
    problems = check_proposal(train, splits)
    problems += [
        f"note: {k} has near-duplicate train task '{t}' (similarity {r:.2f})"
        for k, (t, r) in nearest.items()
        if r >= NEAR_DUPLICATE_SIMILARITY
    ]
    total = len(rows)
    return {
        "seed": seed,
        "config": asdict(cfg),
        "ood_labs": ood_labs,
        "od_tasks": [{"lab_id": lab, "task": task} for lab, task in od_tasks],
        "nearest_train_task": {k: {"task": t, "similarity": round(r, 3)} for k, (t, r) in nearest.items()},
        "stats": {
            "train": split_stats(train, total),
            **{s: split_stats(rs, total) for s, rs in splits.items()},
            "eval_total": split_stats([r for rs in splits.values() for r in rs], total),
            EXCLUDED: split_stats(excluded, total),
        },
        "id_sampling": id_info,
        "problems": problems,
        "per_lab": {
            lab: {s: sum(r["lab_id"] == lab for r in rs) for s, rs in [("train", train), *splits.items()]}
            for lab in sorted({r["lab_id"] for r in rows})
        },
        "episodes": {
            s: [
                {
                    "dataset_repo": r["dataset_repo"],
                    "repo_path": r["repo_path"],
                    "lab_id": r["lab_id"],
                    "robot_name": r["robot_name"],
                    "task": r["task"],
                }
                for r in sorted(rs, key=key)
            ]
            for s, rs in [*splits.items(), (EXCLUDED, excluded)]
        },
    }


def enumerate_proposals(
    rows: list[dict],
    cfg: SplitConfig,
    ood_common: list[str] | None = None,
    ood_custom: list[str] | None = None,
    od_tasks: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], dict]:
    """Cross the top candidates of each holdout type; pinned choices replace the candidate lists."""
    k = cfg.candidates_per_split
    lab_cands = ood_lab_candidates(rows, cfg)
    common = [[lab] for lab in ood_common] if ood_common else [[c["lab_id"]] for c in lab_cands["common"][:k]]
    custom = [[lab] for lab in ood_custom] if ood_custom else [[c["lab_id"]] for c in lab_cands["custom"][:k]]
    common = common or [[]]
    custom = custom or [[]]

    proposals = []
    task_cands_by_ood: dict[tuple, list[dict]] = {}
    for c1, c2 in itertools.product(common, custom):
        ood = c1 + c2
        if od_tasks:
            task_choices = [[t] for t in od_tasks]
        else:
            cands = od_task_candidates(rows, cfg, set(ood))
            task_cands_by_ood[tuple(ood)] = cands
            task_choices = [[(c["lab_id"], c["task"])] for c in cands[:k]] or [[]]
        for tasks in task_choices:
            proposals.append(
                make_proposal(rows, cfg, ood, tasks, cfg.seed + len(proposals))
            )
    for i, p in enumerate(proposals, 1):
        p["proposal_id"] = f"proposal_{i:02d}"
    candidates = {
        "ood_lab": lab_cands,
        "od_task": next(iter(task_cands_by_ood.values()), []),
    }
    return proposals, candidates
