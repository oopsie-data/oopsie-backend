"""Split a proposal's eval episodes into shuffled per-annotator lists.

Usage:
    uv run python -m oopsie_backend.annotation.assign \\
        --proposal eval_split_proposals/proposal_01.json --annotators-file annotators.txt \\
        --copies 2 --split-copies od_task_od_scene=3 --out-dir eval_assignment

Every eval episode goes to ``--copies`` distinct annotators (``--split-copies``
overrides it per split). Annotators get near-equal loads and near-equal
shares of each split, lab and task; each list is shuffled so the same lab
rarely appears twice in a row.

Writes ``annotator_XX.json`` per annotator, ``samples.csv`` mapping each
episode to its annotators, and ``assignment_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from oopsie_backend.annotation.assignment import assign, balance_report, eval_samples
from oopsie_backend.common.repos import DEFAULT_ANNOTATION_DIR
from oopsie_backend.splits.eval_split import SPLITS, TRAIN_ANNOTATED


def _split_copies(spec: str) -> tuple[str, int]:
    split, sep, n = spec.partition("=")
    if not sep or split not in (*SPLITS, TRAIN_ANNOTATED):
        raise argparse.ArgumentTypeError(f"expected SPLIT=N with SPLIT in {(*SPLITS, TRAIN_ANNOTATED)}, got {spec!r}")
    return split, int(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proposal", type=Path, required=True)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--num-annotators", type=int)
    who.add_argument("--annotators", nargs="+", metavar="NAME")
    who.add_argument(
        "--annotators-file", type=Path, metavar="FILE", help="Text file with one annotator name per line"
    )
    parser.add_argument("--copies", type=int, default=2, help="Annotators per episode")
    parser.add_argument(
        "--split-copies", action="append", default=[], type=_split_copies, metavar="SPLIT=N"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ANNOTATION_DIR / "eval_assignment")
    args = parser.parse_args(argv)

    if args.annotators_file:
        annotators = [line.strip() for line in args.annotators_file.read_text().splitlines() if line.strip()]
    elif args.annotators:
        annotators = args.annotators
    else:
        annotators = [f"annotator_{i:02d}" for i in range(1, args.num_annotators + 1)]
    dupes = sorted({a for a in annotators if annotators.count(a) > 1})
    bad = [a for a in annotators if "/" in a or "\\" in a or a.startswith(".")]
    if not annotators or dupes or bad:
        print(f"error: need unique annotator names usable as file names (duplicates {dupes}, invalid {bad})", file=sys.stderr)
        return 1
    proposal = json.loads(args.proposal.read_text())
    samples = eval_samples(proposal)
    copies = {s["split"]: args.copies for s in samples} | dict(args.split_copies)
    try:
        assignment = assign(samples, annotators, copies, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in assignment.items():
        entries = [{"position": i, **r} for i, r in enumerate(items, 1)]
        (args.out_dir / f"{name}.json").write_text(
            json.dumps({"annotator": name, "proposal": proposal.get("proposal_id"), "episodes": entries}, indent=2)
        )

    by_sample: dict[tuple, list[str]] = {}
    for name, items in assignment.items():
        for r in items:
            by_sample.setdefault((r["split"], r["dataset_repo"], r["repo_path"]), []).append(name)
    with (args.out_dir / "samples.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "dataset_repo", "repo_path", "annotators"])
        for (split, repo, path), names in sorted(by_sample.items()):
            w.writerow([split, repo, path, ";".join(sorted(names))])

    report = balance_report(assignment)
    summary = {
        "proposal": proposal.get("proposal_id"),
        "seed": args.seed,
        "copies": copies,
        "samples": len(samples),
        "annotations": sum(report["load"].values()),
        **report,
    }
    (args.out_dir / "assignment_summary.json").write_text(json.dumps(summary, indent=2))

    loads = report["load"].values()
    labs = report["lab_min_max"].values()
    print(
        f"{len(samples)} episodes -> {summary['annotations']} annotations over {len(annotators)} annotators "
        f"(load {min(loads)}-{max(loads)}, worst per-lab spread {max(hi - lo for lo, hi in labs)}, "
        f"shared samples per pair {report['shared_pairs_min_max'][0]}-{report['shared_pairs_min_max'][1]}, "
        f"longest same-lab run {max(report['max_same_lab_run'].values())})",
        file=sys.stderr,
    )
    print(f"wrote {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
