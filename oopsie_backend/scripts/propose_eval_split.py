"""Generate eval split proposals from an episode index.

Usage:
    uv run python -m oopsie_backend.scripts.propose_eval_split \\
        --index episode_index.jsonl --out-dir eval_split_proposals

Three held-out groups per proposal:
  id                 stratified sample whose (lab, robot, cameras, task) stays in train
  od_task_id_scene   one full task from a medium multi-task lab
  od_task_od_scene   two full small labs, one on a common and one on a custom robot

Without pins, the top ``--candidates`` of each holdout type are crossed.
Repeated pins are alternatives: each value yields its own proposals.
Writes ``REPORT.md``, ``candidates.json`` and one ``proposal_XX.json`` per proposal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.utils.episode_index import normalize_task, read_index
from oopsie_backend.utils.eval_split import (
    EXCLUDED,
    SPLITS,
    SplitConfig,
    enumerate_proposals,
    lab_summary,
)


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
            "| lab | episodes | share | robots | tasks | tasks shared with other labs |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for c in cands:
            lines.append(
                f"| {c['lab_id']} | {c['episodes']} | {_pct(c['frac'])} | {', '.join(c['robots'])} | "
                f"{c['tasks']} | {len(c['shared_tasks'])} |"
            )
        lines.append("")
    lines += ["### Held-out task candidates", ""]
    if not candidates["od_task"]:
        return lines + ["None satisfy the constraints.", ""]
    lines += [
        "| lab | task | episodes | share of lab | lab tasks | success | nearest other task | similarity |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for c in candidates["od_task"]:
        lines.append(
            f"| {c['lab_id']} | {c['task']} | {c['episodes']} | {_pct(c['lab_share'])} | "
            f"{c['lab_tasks']} | {_pct(c['success_rate'])} | {c['nearest_task']} | {c['similarity']:.2f} |"
        )
    return lines + [""]


def _proposal_section(p: dict) -> list[str]:
    tasks = "; ".join(f"{t['lab_id']} :: {t['task']}" for t in p["od_tasks"]) or "-"
    lines = [
        f"## {p['proposal_id']}",
        "",
        f"- held-out labs: {', '.join(p['ood_labs']) or '-'}",
        f"- held-out task: {tasks}",
    ]
    for k, v in p["nearest_train_task"].items():
        lines.append(f"- nearest train task to `{k}`: `{v['task']}` (similarity {v['similarity']})")
    info = p["id_sampling"]
    lines += [
        f"- ID sampling: {info['sampled_groups']}/{info['eligible_groups']} eligible groups sampled "
        f"({info['groups']} total, {info['ineligible_episodes']} episodes in groups too small to sample)",
        "",
        "| split | episodes | share | hours | labs | robots | tasks | success | failure desc. | ≥2 annotators |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", *SPLITS, "eval_total", EXCLUDED):
        s = p["stats"][name]
        lines.append(
            f"| {name} | {s['episodes']} | {_pct(s['frac'])} | {s['hours']:.1f} | {s['labs']} | "
            f"{s['robots']} | {s['tasks']} | {_pct(s['success_rate'])} | "
            f"{s['with_failure_description']} | {s['multi_annotator']} |"
        )
    lines.append("")
    if p["problems"]:
        lines += ["Checks:", ""] + [f"- {msg}" for msg in p["problems"]] + [""]
    else:
        lines += ["Checks: none", ""]
    return lines


def write_report(
    out_dir: Path, rows: list[dict], cfg: SplitConfig, proposals: list[dict], candidates: dict
) -> Path:
    lines = [
        "# Eval split proposals",
        "",
        f"{len(rows)} episodes, {len({r['lab_id'] for r in rows})} labs. "
        "Scene = (lab, robot, cameras); task = normalized instruction.",
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
        "| proposal | held-out labs | held-out task | eval episodes | eval share | checks |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in proposals:
        s = p["stats"]["eval_total"]
        tasks = "; ".join(f"{t['lab_id']} :: {t['task']}" for t in p["od_tasks"]) or "-"
        n_problems = sum(not m.startswith("note:") for m in p["problems"])
        lines.append(
            f"| {p['proposal_id']} | {', '.join(p['ood_labs']) or '-'} | {tasks} | "
            f"{s['episodes']} | {_pct(s['frac'])} | {n_problems} |"
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
    pins = parser.add_argument_group("pins (repeatable; replace the candidate search)")
    pins.add_argument("--ood-common", action="append", default=[], metavar="LAB")
    pins.add_argument("--ood-custom", action="append", default=[], metavar="LAB")
    pins.add_argument("--od-task", action="append", default=[], type=_parse_task, metavar="LAB::task")
    pins.add_argument("--common-robot", action="append", default=[], metavar="ROBOT")
    pins.add_argument("--custom-robot", action="append", default=[], metavar="ROBOT")
    knobs = parser.add_argument_group("sizes")
    knobs.add_argument("--id-frac", type=float, default=d.id_frac)
    knobs.add_argument("--id-alpha", type=float, default=d.id_alpha, help="group weight = size**alpha")
    knobs.add_argument("--id-max-group-frac", type=float, default=d.id_max_group_frac)
    knobs.add_argument("--id-min-group", type=int, default=d.id_min_group)
    knobs.add_argument("--ood-lab-min", type=int, default=d.ood_lab_min)
    knobs.add_argument("--ood-lab-max-frac", type=float, default=d.ood_lab_max_frac)
    knobs.add_argument("--od-task-lab-min-frac", type=float, default=d.od_task_lab_min_frac)
    knobs.add_argument("--od-task-lab-max-frac", type=float, default=d.od_task_lab_max_frac)
    knobs.add_argument("--od-task-min-tasks", type=int, default=d.od_task_min_tasks)
    knobs.add_argument("--od-task-min-episodes", type=int, default=d.od_task_min_episodes)
    knobs.add_argument(
        "--holdout-cap",
        type=int,
        default=d.holdout_cap,
        help="Max eval episodes per held-out lab or task; the rest are excluded from train and eval.",
    )
    knobs.add_argument("--od-task-max-share", type=float, default=d.od_task_max_share)
    args = parser.parse_args(argv)

    cfg = SplitConfig(
        id_frac=args.id_frac,
        id_alpha=args.id_alpha,
        id_max_group_frac=args.id_max_group_frac,
        id_min_group=args.id_min_group,
        ood_lab_min=args.ood_lab_min,
        ood_lab_max_frac=args.ood_lab_max_frac,
        od_task_lab_min_frac=args.od_task_lab_min_frac,
        od_task_lab_max_frac=args.od_task_lab_max_frac,
        od_task_min_tasks=args.od_task_min_tasks,
        od_task_min_episodes=args.od_task_min_episodes,
        od_task_max_share=args.od_task_max_share,
        holdout_cap=args.holdout_cap,
        candidates_per_split=args.candidates,
        common_robots=args.common_robot,
        custom_robots=args.custom_robot,
        seed=args.seed,
    )
    rows = read_index(args.index)
    if not rows:
        print(f"error: {args.index} is empty", file=sys.stderr)
        return 1
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
