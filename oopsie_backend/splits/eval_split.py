"""Propose eval splits (ID, OD-task/ID-scene, OD-task/OD-scene) from an episode index.

Robots are compared by embodiment, the alias-resolved ``robot`` field set by
``apply_robot_aliases``. A scene is approximated by ``(lab_id, robot, cameras)``; a task by the
normalized language instruction. An annotation budget is split across the
three groups; held-out labs and tasks are ranked by how close their size is to
their share of the budget, and the part of a held-out unit beyond its share is
excluded from both train and eval. Every proposal is checked against the
remaining training pool so violations show up in the report instead of being
silently accepted.
"""

from __future__ import annotations

import difflib
import itertools
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

NEAR_DUPLICATE_SIMILARITY = 0.85
EXCLUDED = "excluded"
ID_SPLIT = "id"
OD_TASK_SPLIT = "od_task_id_scene"
OOD_LAB_SPLIT = "od_task_od_scene"
SPLITS = (ID_SPLIT, OD_TASK_SPLIT, OOD_LAB_SPLIT)
TRAIN_ANNOTATED = "train_annotated"


@dataclass
class SplitConfig:
    eval_budget: int | None = 2000
    budget_frac: float = 0.05
    split_weights: tuple[float, float, float] = (0.2, 0.4, 0.4)
    id_alpha: float = 0.5
    id_max_group_frac: float = 0.25
    id_min_group: int = 8
    ood_lab_min: int = 40
    ood_lab_max_quantile: float = 0.5
    od_task_lab_quantiles: tuple[float, float] = (0.25, 0.75)
    od_task_min_tasks: int = 3
    od_task_min_episodes: int = 25
    od_task_max_share: float = 0.4
    cap_holdouts: bool = True
    holdout_task_cap: int | None = 50
    num_od_tasks: int = 3
    num_ood_common: int = 1
    num_ood_custom: int = 1
    candidates_per_split: int = 3
    common_robots: list[str] = field(default_factory=list)
    custom_robots: list[str] = field(default_factory=list)
    train_only_labs: list[str] = field(default_factory=list)
    seed: int = 0


def budget(cfg: SplitConfig, total: int) -> dict[str, int]:
    """Episodes to label per split."""
    n = cfg.eval_budget if cfg.eval_budget is not None else round(cfg.budget_frac * total)
    norm = sum(cfg.split_weights)
    return {s: round(n * w / norm) for s, w in zip(SPLITS, cfg.split_weights)}


def unit_targets(cfg: SplitConfig, total: int) -> dict[str, int]:
    """Target size of one held-out lab and of one held-out task."""
    b = budget(cfg, total)
    n_labs = max(1, cfg.num_ood_common + cfg.num_ood_custom)
    task_target = b[OD_TASK_SPLIT] // max(1, cfg.num_od_tasks)
    if cfg.holdout_task_cap:
        task_target = min(task_target, cfg.holdout_task_cap)
    return {
        OOD_LAB_SPLIT: max(1, b[OOD_LAB_SPLIT] // n_labs),
        OD_TASK_SPLIT: max(1, task_target),
    }


def labelable(task_counts: Counter, cfg: SplitConfig) -> int:
    """Episodes of a held-out unit left after capping each of its tasks at ``holdout_task_cap``."""
    if not cfg.holdout_task_cap:
        return sum(task_counts.values())
    return sum(min(n, cfg.holdout_task_cap) for n in task_counts.values())


def _size_score(n: int, target: int) -> float:
    return abs(math.log(n / target))


def _lab_size_quantiles(rows: list[dict]) -> dict[str, float]:
    """Rank of each lab by episode count, scaled to [0, 1] (0 = smallest)."""
    sizes = sorted(Counter(r["lab_id"] for r in rows).items(), key=lambda kv: (kv[1], kv[0]))
    denom = max(1, len(sizes) - 1)
    return {lab: i / denom for i, (lab, _) in enumerate(sizes)}


DEFAULT_ROBOT_ALIASES = Path(__file__).resolve().parents[2] / "configs" / "robot_aliases.yaml"


def canonical_robot(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def load_robot_aliases(path: Path | None = DEFAULT_ROBOT_ALIASES) -> dict[str, str]:
    """Map normalized raw robot name -> embodiment from a ``{embodiment: [raw names]}`` YAML file."""
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    aliases: dict[str, str] = {}
    for embodiment, names in raw.items():
        for name in [embodiment, *(names or [])]:
            norm = canonical_robot(str(name))
            if aliases.get(norm, embodiment) != embodiment:
                raise ValueError(f"{path}: {name!r} listed under {aliases[norm]!r} and {embodiment!r}")
            aliases[norm] = str(embodiment)
    return aliases


def resolve_robot(name: str, aliases: dict[str, str]) -> str:
    norm = canonical_robot(name)
    return aliases.get(norm, norm)


def apply_robot_aliases(rows: list[dict], aliases: dict[str, str]) -> list[str]:
    """Set ``row["robot"]`` on every row; return raw names absent from ``aliases``."""
    unmapped = set()
    for r in rows:
        r["robot"] = resolve_robot(r["robot_name"], aliases)
        if canonical_robot(r["robot_name"]) not in aliases:
            unmapped.add(r["robot_name"])
    return sorted(unmapped)


def key(row: dict) -> tuple[str, str]:
    return row["dataset_repo"], row["repo_path"]


def scene(row: dict) -> tuple[str, str, str]:
    return row["lab_id"], row["robot"], row["cameras"]


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
    """``common`` for embodiments used by at least two labs, ``custom`` otherwise; config overrides win."""
    labs_per_robot: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        labs_per_robot[r["robot"]].add(r["lab_id"])
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
        robots = sorted({r["robot"] for r in rs})
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
    """Labs in the lower size quantiles, split by embodiment class, best first.

    A lab's size is its ``labelable`` episode count. The score is the distance
    of that size to the per-lab target plus ``log(episodes / labelable)``, the
    share of the lab that the task cap would drop from both train and eval.
    """
    target = unit_targets(cfg, len(rows))[OOD_LAB_SPLIT]
    quantiles = _lab_size_quantiles(rows)
    by_lab = _by(rows, "lab_id")
    tasks_by_lab = {lab: {r["task"] for r in rs} for lab, rs in by_lab.items()}
    out: dict[str, list[dict]] = {"common": [], "custom": []}
    summary = lab_summary(rows, cfg)
    for lab in summary:
        lab["labelable"] = labelable(Counter(r["task"] for r in by_lab[lab["lab_id"]]), cfg)
        lab["score"] = _size_score(lab["labelable"], target) + math.log(lab["episodes"] / lab["labelable"])
    for lab in sorted(summary, key=lambda s: s["score"]):
        if (
            lab["episodes"] < cfg.ood_lab_min
            or quantiles[lab["lab_id"]] > cfg.ood_lab_max_quantile
            or lab["lab_id"] in cfg.train_only_labs
        ):
            continue
        lab["target"] = target
        other_tasks = set().union(
            *(t for other, t in tasks_by_lab.items() if other != lab["lab_id"])
        )
        lab["shared_tasks"] = sorted(tasks_by_lab[lab["lab_id"]] & other_tasks)
        out[lab["embodiment"]].append(lab)
    return out


def _jaccard(wa: set[str], wb: set[str]) -> float:
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def task_similarity(a: str, b: str) -> float:
    """Max of character-level ``difflib`` ratio and word-set Jaccard, so word swaps count as similar."""
    return max(difflib.SequenceMatcher(None, a, b).ratio(), _jaccard(set(a.split()), set(b.split())))


def nearest_task(task: str, pool: set[str]) -> tuple[str, float]:
    """Most similar task string in ``pool`` by ``task_similarity``.

    The exact ``difflib`` ratio is computed only when its length and
    character-count bounds can beat the best similarity so far.
    """
    words = set(task.split())
    m = difflib.SequenceMatcher(None, b=task)
    best, best_sim = "", 0.0
    for other in sorted(pool):
        sim = _jaccard(words, set(other.split()))
        floor = max(best_sim, sim)
        if 2 * min(len(task), len(other)) / (len(task) + len(other)) > floor:
            m.set_seq1(other)
            if m.quick_ratio() > floor:
                sim = max(sim, m.ratio())
        if sim > best_sim:
            best, best_sim = other, sim
    return best, best_sim


def od_task_candidates(rows: list[dict], cfg: SplitConfig) -> list[dict]:
    """Tasks of medium multi-task labs that can be held out with their scenes kept in training.

    Medium labs lie within ``od_task_lab_quantiles`` of the lab size ranking.
    A task qualifies when it is absent from every other lab, has no
    near-duplicate task anywhere, holds between ``od_task_min_episodes``
    episodes and ``od_task_max_share`` of its lab, and each of its scenes still
    has training episodes on other tasks. Candidates are ordered by closeness
    to the task target size, with each lab's best task first so that the top
    of the list spans several labs.
    """
    target = unit_targets(cfg, len(rows))[OD_TASK_SPLIT]
    quantiles = _lab_size_quantiles(rows)
    lo, hi = cfg.od_task_lab_quantiles
    all_tasks = {r["task"] for r in rows}
    labs_per_task: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        labs_per_task[r["task"]].add(r["lab_id"])
    out = []
    for lab, rs in _by(rows, "lab_id").items():
        if not lo <= quantiles[lab] <= hi or lab in cfg.train_only_labs:
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
            if similarity >= NEAR_DUPLICATE_SIMILARITY:
                continue
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
                    "target": target,
                }
            )
    out.sort(key=lambda c: _size_score(c["episodes"], c["target"]))
    rank_in_lab: Counter = Counter()
    for c in out:
        c["rank_in_lab"] = rank_in_lab[c["lab_id"]]
        rank_in_lab[c["lab_id"]] += 1
    out.sort(key=lambda c: (c["rank_in_lab"], _size_score(c["episodes"], c["target"])))
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


def _cap_units(
    rs: list[dict], unit_fn, cap: int | dict, alpha: float, rng: random.Random, excluded: list[dict]
) -> list[dict]:
    """Subsample each unit of ``rs`` to its cap, appending the rest to ``excluded``.

    ``cap`` is one limit for every unit or a limit per unit.
    """
    kept = []
    for unit in sorted({unit_fn(r) for r in rs}):
        unit_rs = [r for r in rs if unit_fn(r) == unit]
        limit = cap[unit] if isinstance(cap, dict) else cap
        if len(unit_rs) > limit:
            sub = _subsample(unit_rs, limit, alpha, rng)
            sub_keys = {key(r) for r in sub}
            excluded.extend(r for r in unit_rs if key(r) not in sub_keys)
            unit_rs = sub
        kept.extend(unit_rs)
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
    eligible = {
        g: rs for g, rs in groups.items() if len(rs) >= cfg.id_min_group and g[0] not in cfg.train_only_labs
    }
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
    train_robots = {r["robot"] for r in train}

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
    gone = {r["robot"] for r in splits[OOD_LAB_SPLIT]} - train_robots
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
        "robots": len({r["robot"] for r in rows}),
        "tasks": len({r["task"] for r in rows}),
        "success_rate": _success_rate(rows),
        "with_failure_description": sum(r["has_failure_description"] for r in rows),
        "multi_annotator": sum(len(r["annotators"]) >= 2 for r in rows),
    }


def _per_lab_counts(parts: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    counts = {name: Counter(r["lab_id"] for r in rs) for name, rs in parts.items()}
    labs = sorted(set().union(*counts.values()))
    return {lab: {name: c[lab] for name, c in counts.items()} for lab in labs}


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
    if cfg.cap_holdouts:
        task_of = lambda r: (r["lab_id"], r["task"])
        for name, field_fn in ((OOD_LAB_SPLIT, lambda r: r["lab_id"]), (OD_TASK_SPLIT, task_of)):
            if cfg.holdout_task_cap:
                splits[name] = _cap_units(splits[name], task_of, cfg.holdout_task_cap, cfg.id_alpha, rng, excluded)
            sizes = Counter(field_fn(r) for r in splits[name])
            caps = _allocate(sizes, budget(cfg, len(rows))[name], 0.0, 1.0)
            splits[name] = _cap_units(splits[name], field_fn, caps, cfg.id_alpha, rng, excluded)
    targets = budget(cfg, len(rows))
    splits[ID_SPLIT], id_info = sample_id(pool, targets[ID_SPLIT], cfg, rng)
    id_keys = {key(r) for r in splits[ID_SPLIT]}
    train = [r for r in pool if key(r) not in id_keys]

    train_tasks = {r["task"] for r in train}
    nearest = {
        f"{lab}::{task}": nearest_task(task, train_tasks) for lab, task in od_tasks
    }
    problems = check_proposal(train, splits)
    in_eval = sorted({r["lab_id"] for s in SPLITS for r in splits[s]} & set(cfg.train_only_labs))
    if in_eval:
        problems.append(f"train-only lab(s) in eval: {in_eval}")
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
        "budget": {"total": total, "eval_target": sum(targets.values()), **targets},
        "stats": {
            "train": split_stats(train, total),
            **{s: split_stats(rs, total) for s, rs in splits.items()},
            "eval_total": split_stats([r for rs in splits.values() for r in rs], total),
            EXCLUDED: split_stats(excluded, total),
        },
        "id_sampling": id_info,
        "problems": problems,
        "per_lab": _per_lab_counts({"train": train, **splits, EXCLUDED: excluded}),
        "episodes": {
            s: [
                {
                    "dataset_repo": r["dataset_repo"],
                    "repo_path": r["repo_path"],
                    "lab_id": r["lab_id"],
                    "robot_name": r["robot_name"],
                    "robot": r["robot"],
                    "task": r["task"],
                }
                for r in sorted(rs, key=key)
            ]
            for s, rs in [*splits.items(), (EXCLUDED, excluded)]
        },
    }


def train_pool(rows: list[dict], proposal: dict) -> list[dict]:
    """Rows a proposal leaves in train, minus anything from its held-out labs or tasks."""
    taken = {(e["dataset_repo"], e["repo_path"]) for rs in proposal["episodes"].values() for e in rs}
    ood_labs = set(proposal["ood_labs"])
    od_tasks = {t["task"] for t in proposal["od_tasks"]}
    return [
        r for r in rows
        if key(r) not in taken and r["lab_id"] not in ood_labs and r["task"] not in od_tasks
    ]


def _even_split(sizes: dict, n: int, rng: random.Random) -> dict:
    """Split ``n`` equally across groups, capped at each group's size; leftovers go to random groups."""
    alloc = {g: 0 for g in sizes}
    order = sorted(sizes)
    rng.shuffle(order)
    while n > 0:
        open_groups = [g for g in order if alloc[g] < sizes[g]]
        if not open_groups:
            break
        share = max(1, n // len(open_groups))
        for g in open_groups:
            take = min(share, sizes[g] - alloc[g], n)
            alloc[g] += take
            n -= take
            if not n:
                break
    return alloc


def coverage_sample(rows: list[dict], n: int, levels: list, rng: random.Random) -> list[dict]:
    """Pick ``n`` rows spread evenly over nested strata, ignoring stratum sizes.

    ``levels`` are functions of a row. The budget is split equally over the
    values of the first level, then recursively within each value; rows are
    drawn at random once the levels run out.
    """
    if n >= len(rows):
        return list(rows)
    if not levels:
        return rng.sample(sorted(rows, key=key), n)
    groups: dict = defaultdict(list)
    for r in rows:
        groups[levels[0](r)].append(r)
    alloc = _even_split({g: len(rs) for g, rs in groups.items()}, n, rng)
    picked = []
    for g in sorted(groups):
        picked.extend(coverage_sample(groups[g], alloc[g], levels[1:], rng))
    return picked


def _ranked_sets(cands: list[dict], n: int, k: int) -> list[list[dict]]:
    """The ``k`` best ``n``-subsets of ``cands`` by summed score; ``cands`` is sorted best first."""
    if n <= 0:
        return [[]]
    pool = cands[: n + k - 1]
    if len(pool) < n:
        return [pool]
    combos = itertools.combinations(pool, n)
    score = lambda x: x.get("score", _size_score(x["episodes"], x["target"]))
    ranked = sorted(combos, key=lambda c: sum(score(x) for x in c))
    return [list(c) for c in ranked[:k]]


def enumerate_proposals(
    rows: list[dict],
    cfg: SplitConfig,
    ood_common: list[str] | None = None,
    ood_custom: list[str] | None = None,
    od_tasks: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], dict]:
    """Cross the top candidate sets of each holdout type.

    Held-out labs are chosen as ``num_ood_common`` + ``num_ood_custom`` labs,
    held-out tasks as ``num_od_tasks`` tasks from distinct labs outside the
    held-out labs. A pinned list replaces the search for its holdout type.
    """
    k = cfg.candidates_per_split
    lab_cands = ood_lab_candidates(rows, cfg)
    task_cands = od_task_candidates(rows, cfg)
    common = [ood_common] if ood_common else [
        [c["lab_id"] for c in s] for s in _ranked_sets(lab_cands["common"], cfg.num_ood_common, k)
    ]
    custom = [ood_custom] if ood_custom else [
        [c["lab_id"] for c in s] for s in _ranked_sets(lab_cands["custom"], cfg.num_ood_custom, k)
    ]

    proposals = []
    for c1, c2 in itertools.product(common, custom):
        ood = c1 + c2
        if od_tasks:
            task_sets = [od_tasks]
        else:
            per_lab_best = [
                c for c in task_cands if c["rank_in_lab"] == 0 and c["lab_id"] not in ood
            ]
            task_sets = [
                [(c["lab_id"], c["task"]) for c in s]
                for s in _ranked_sets(per_lab_best, cfg.num_od_tasks, k)
            ]
        for tasks in task_sets:
            proposals.append(
                make_proposal(rows, cfg, ood, tasks, cfg.seed + len(proposals))
            )
    for i, p in enumerate(proposals, 1):
        p["proposal_id"] = f"proposal_{i:02d}"
    return proposals, {"ood_lab": lab_cands, "od_task": task_cands}
