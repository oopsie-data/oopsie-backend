"""Generate eval split proposals from an episode index.

Usage:
    uv run python -m oopsie_backend.splits.propose \\
        --index episode_index.jsonl --out-dir eval_split_proposals

Three held-out groups per proposal:
  id                 stratified sample whose (lab, robot, cameras, task) stays in train
  od_task_id_scene   full tasks from several medium multi-task labs, one per lab
  od_task_od_scene   full small labs on common and on custom robots

The annotation budget (``--eval-budget`` or ``--budget-frac`` of all episodes)
is split by ``--split-weights``. Held-out labs and tasks are ranked by how
close their size is to their share; episodes of a held-out unit beyond that
share are excluded from both train and eval.

Without pins, the top ``--candidates`` sets of each holdout type are crossed.
Repeated pins of one kind form a single set that replaces the search.
Writes ``REPORT.md``, ``candidates.json`` and one ``proposal_XX.json`` per proposal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.splits.eval_split import (
    DEFAULT_ROBOT_ALIASES,
    EXCLUDED,
    SPLITS,
    SplitConfig,
    apply_robot_aliases,
    budget,
    enumerate_proposals,
    lab_summary,
    load_robot_aliases,
    resolve_robot,
    unit_targets,
)
from oopsie_backend.splits.planning import normalize_task, read_index


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _lab_table(summary: list[dict]) -> list[str]:
    lines = [
        "| lab | episodes | share | hours | robots | embodiment | scenes | tasks | success |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['lab_id']} | {s['episodes']} | {_pct(s['frac'])} | {s['hours']:.1f} | "
            f"{', '.join(s['robots'])} | {s['embodiment']} | {s['scenes']} | {s['tasks']} | "
            f"{_pct(s['success_rate'])} |"
        )
    return lines


def _candidate_tables(candidates: dict) -> list[str]:
    lines = []
    for kind in ("common", "custom"):
        lines += [f"### Held-out lab candidates ({kind} embodiment)", ""]
        cands = candidates["ood_lab"][kind]
        if not cands:
            lines += ["None within the size limits.", ""]
            continue
        lines += [
            "| lab | episodes | target | share | robots | tasks | tasks shared with other labs |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
        for c in cands:
            lines.append(
                f"| {c['lab_id']} | {c['episodes']} | {c['target']} | {_pct(c['frac'])} | {', '.join(c['robots'])} | "
                f"{c['tasks']} | {len(c['shared_tasks'])} |"
            )
        lines.append("")
    lines += ["### Held-out task candidates", "", "Each lab's best task is listed before any lab's second.", ""]
    if not candidates["od_task"]:
        return lines + ["None satisfy the constraints.", ""]
    lines += [
        "| lab | task | episodes | target | share of lab | lab tasks | success | nearest other task | similarity |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for c in candidates["od_task"]:
        lines.append(
            f"| {c['lab_id']} | {c['task']} | {c['episodes']} | {c['target']} | {_pct(c['lab_share'])} | "
            f"{c['lab_tasks']} | {_pct(c['success_rate'])} | {c['nearest_task']} | {c['similarity']:.2f} |"
        )
    return lines + [""]


def _proposal_section(p: dict) -> list[str]:
    tasks = "; ".join(f"{t['lab_id']} :: {t['task']}" for t in p["od_tasks"]) or "-"
    lines = [
        f"## {p['proposal_id']}",
        "",
        f"- held-out labs: {', '.join(p['ood_labs']) or '-'}",
        f"- held-out tasks: {tasks}",
    ]
    for k, v in p["nearest_train_task"].items():
        lines.append(f"- nearest train task to `{k}`: `{v['task']}` (similarity {v['similarity']})")
    info = p["id_sampling"]
    lines += [
        f"- ID sampling: {info['sampled_groups']}/{info['eligible_groups']} eligible groups sampled "
        f"({info['groups']} total, {info['ineligible_episodes']} episodes in groups too small to sample)",
        "",
        "| split | episodes | target | share | hours | labs | robots | tasks | success | failure desc. | ≥2 annotators |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    targets = {**p["budget"], "eval_total": p["budget"]["eval_target"]}
    for name in ("train", *SPLITS, "eval_total", EXCLUDED):
        s = p["stats"][name]
        lines.append(
            f"| {name} | {s['episodes']} | {targets.get(name, '-')} | {_pct(s['frac'])} | {s['hours']:.1f} | {s['labs']} | "
            f"{s['robots']} | {s['tasks']} | {_pct(s['success_rate'])} | "
            f"{s['with_failure_description']} | {s['multi_annotator']} |"
        )
    lines.append("")
    if p["problems"]:
        lines += ["Checks:", ""] + [f"- {msg}" for msg in p["problems"]] + [""]
    else:
        lines += ["Checks: none", ""]
    return lines


def _budget_lines(rows: list[dict], cfg: SplitConfig) -> list[str]:
    b = budget(cfg, len(rows))
    u = unit_targets(cfg, len(rows))
    n = sum(b.values())
    return [
        f"- available: {len(rows)} episodes; {sum(r['success'] is not None for r in rows)} with a "
        f"success label, {sum(r['has_failure_description'] for r in rows)} with a failure description, "
        f"{sum(len(r['annotators']) >= 2 for r in rows)} with ≥2 annotators",
        f"- eval budget: {n} episodes ({_pct(n / len(rows))})",
        "- per split: " + ", ".join(f"{s} {b[s]}" for s in SPLITS),
        f"- {cfg.num_ood_common} common + {cfg.num_ood_custom} custom held-out lab(s), target {u['od_task_od_scene']} each; "
        f"{cfg.num_od_tasks} held-out task(s), target {u['od_task_id_scene']} each",
    ]


def write_report(
    out_dir: Path, rows: list[dict], cfg: SplitConfig, proposals: list[dict], candidates: dict
) -> Path:
    lines = [
        "# Eval split proposals",
        "",
        f"{len(rows)} episodes, {len({r['lab_id'] for r in rows})} labs. "
        "Robot = embodiment after aliasing; scene = (lab, robot, cameras); task = normalized instruction.",
        "",
        "## Budget",
        "",
        *_budget_lines(rows, cfg),
        "",
        "## Labs",
        "",
        *_lab_table(lab_summary(rows, cfg)),
        "",
        "## Candidates",
        "",
        *_candidate_tables(candidates),
        "## Summary",
        "",
        "| proposal | held-out labs | held-out tasks | eval episodes | eval share | excluded | train share | checks |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in proposals:
        s = p["stats"]["eval_total"]
        tasks = "; ".join(f"{t['lab_id']} :: {t['task']}" for t in p["od_tasks"]) or "-"
        n_problems = sum(not m.startswith("note:") for m in p["problems"])
        lines.append(
            f"| {p['proposal_id']} | {', '.join(p['ood_labs']) or '-'} | {tasks} | "
            f"{s['episodes']} | {_pct(s['frac'])} | {p['stats'][EXCLUDED]['episodes']} | "
            f"{_pct(p['stats']['train']['frac'])} | {n_problems} |"
        )
    lines.append("")
    for p in proposals:
        lines += _proposal_section(p)
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_task(spec: str) -> tuple[str, str]:
    lab, sep, task = spec.partition("::")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected LAB::task, got {spec!r}")
    return lab.strip(), normalize_task(task)


def main(argv: list[str] | None = None) -> int:
    d = SplitConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_split_proposals"))
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--candidates", type=int, default=d.candidates_per_split)
    pins = parser.add_argument_group("pins (repeatable; all values are held out together)")
    pins.add_argument("--ood-common", action="append", default=[], metavar="LAB")
    pins.add_argument("--ood-custom", action="append", default=[], metavar="LAB")
    pins.add_argument("--od-task", action="append", default=[], type=_parse_task, metavar="LAB::task")
    parser.add_argument(
        "--robot-aliases",
        type=Path,
        default=DEFAULT_ROBOT_ALIASES,
        help="YAML of embodiment -> raw robot names",
    )
    pins.add_argument("--common-robot", action="append", default=[], metavar="ROBOT")
    pins.add_argument("--custom-robot", action="append", default=[], metavar="ROBOT")
    knobs = parser.add_argument_group("sizes")
    knobs.add_argument("--num-od-tasks", type=int, default=d.num_od_tasks, help="Held-out tasks, from distinct labs")
    knobs.add_argument("--num-ood-common", type=int, default=d.num_ood_common, help="Held-out labs on common robots")
    knobs.add_argument("--num-ood-custom", type=int, default=d.num_ood_custom, help="Held-out labs on custom robots")
    knobs.add_argument("--eval-budget", type=int, default=d.eval_budget, help="Episodes to label in total; -1 uses --budget-frac")
    knobs.add_argument("--budget-frac", type=float, default=d.budget_frac, help="Used when --eval-budget is -1")
    knobs.add_argument(
        "--split-weights",
        type=lambda v: tuple(float(x) for x in v.split(",")),
        default=d.split_weights,
        metavar="ID,OD_TASK,OOD_LAB",
    )
    knobs.add_argument("--id-alpha", type=float, default=d.id_alpha, help="group weight = size**alpha")
    knobs.add_argument("--id-max-group-frac", type=float, default=d.id_max_group_frac)
    knobs.add_argument("--id-min-group", type=int, default=d.id_min_group)
    knobs.add_argument("--ood-lab-min", type=int, default=d.ood_lab_min)
    knobs.add_argument(
        "--ood-lab-max-quantile",
        type=float,
        default=d.ood_lab_max_quantile,
        help="Held-out labs come from labs up to this size quantile",
    )
    knobs.add_argument(
        "--od-task-lab-quantiles",
        type=lambda v: tuple(float(x) for x in v.split(",")),
        default=d.od_task_lab_quantiles,
        metavar="LO,HI",
        help="Size-quantile window of labs that may lose a task",
    )
    knobs.add_argument("--od-task-min-tasks", type=int, default=d.od_task_min_tasks)
    knobs.add_argument("--od-task-min-episodes", type=int, default=d.od_task_min_episodes)
    knobs.add_argument(
        "--no-holdout-cap",
        action="store_true",
        help="Label held-out labs and tasks in full instead of capping them at their budget share",
    )
    knobs.add_argument("--od-task-max-share", type=float, default=d.od_task_max_share)
    args = parser.parse_args(argv)

    cfg = SplitConfig(
        eval_budget=args.eval_budget if args.eval_budget >= 0 else None,
        budget_frac=args.budget_frac,
        split_weights=args.split_weights,
        id_alpha=args.id_alpha,
        id_max_group_frac=args.id_max_group_frac,
        id_min_group=args.id_min_group,
        ood_lab_min=args.ood_lab_min,
        ood_lab_max_quantile=args.ood_lab_max_quantile,
        od_task_lab_quantiles=args.od_task_lab_quantiles,
        od_task_min_tasks=args.od_task_min_tasks,
        od_task_min_episodes=args.od_task_min_episodes,
        od_task_max_share=args.od_task_max_share,
        cap_holdouts=not args.no_holdout_cap,
        num_od_tasks=args.num_od_tasks,
        num_ood_common=args.num_ood_common,
        num_ood_custom=args.num_ood_custom,
        candidates_per_split=args.candidates,
        common_robots=args.common_robot,
        custom_robots=args.custom_robot,
        seed=args.seed,
    )
    rows = read_index(args.index)
    if not rows:
        print(f"error: {args.index} is empty", file=sys.stderr)
        return 1
    aliases = load_robot_aliases(args.robot_aliases)
    unmapped = apply_robot_aliases(rows, aliases)
    if unmapped:
        print(f"robot names not in {args.robot_aliases}: {unmapped}", file=sys.stderr)
    cfg.common_robots = [resolve_robot(r, aliases) for r in cfg.common_robots]
    cfg.custom_robots = [resolve_robot(r, aliases) for r in cfg.custom_robots]
    proposals, candidates = enumerate_proposals(
        rows, cfg, args.ood_common, args.ood_custom, args.od_task
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidates.json").write_text(json.dumps(candidates, indent=2))
    for p in proposals:
        (args.out_dir / f"{p['proposal_id']}.json").write_text(json.dumps(p, indent=2))
    report = write_report(args.out_dir, rows, cfg, proposals, candidates)
    print(f"wrote {len(proposals)} proposal(s) and {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
