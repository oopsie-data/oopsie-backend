"""Sample train episodes for extra annotation, spread evenly across labs.

Usage:
    uv run python -m oopsie_backend.splits.sample_train \\
        --index episode_index.jsonl --proposal eval_split_proposals/proposal_02.json \\
        --num-episodes 1000 --out train_annotation_sample.json

The pool is the proposal's train set: its eval and excluded episodes, and
every episode of a held-out lab or task, are removed. The budget is split
equally over labs, then within each lab over scenes (robot, cameras), tasks,
policies and success labels, so small labs and rare tasks are covered instead
of following dataset proportions.

The output has the same ``episodes`` layout as a proposal, with a single
``train_annotated`` split, and can be passed to ``assign_eval_annotators``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from oopsie_backend.common.repos import DEFAULT_ANNOTATION_DIR
from oopsie_backend.splits.eval_split import (
    DEFAULT_ROBOT_ALIASES,
    TRAIN_ANNOTATED,
    apply_robot_aliases,
    coverage_sample,
    key,
    load_robot_aliases,
    split_stats,
    train_pool,
)
from oopsie_backend.splits.planning import read_index

LEVELS = [
    lambda r: r["lab_id"],
    lambda r: (r["robot"], r["cameras"]),
    lambda r: r["task"],
    lambda r: str(r["policy_name"]),
    lambda r: str(r["success"]),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--robot-aliases", type=Path, default=DEFAULT_ROBOT_ALIASES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_ANNOTATION_DIR / "train_annotation_sample.json")
    args = parser.parse_args(argv)

    rows = read_index(args.index)
    apply_robot_aliases(rows, load_robot_aliases(args.robot_aliases))
    proposal = json.loads(args.proposal.read_text())
    pool = train_pool(rows, proposal)
    picked = coverage_sample(pool, args.num_episodes, LEVELS, random.Random(args.seed))

    per_lab = {lab: sum(r["lab_id"] == lab for r in picked) for lab in sorted({r["lab_id"] for r in pool})}
    out = {
        "proposal_id": f"{proposal['proposal_id']}_{TRAIN_ANNOTATED}",
        "source_proposal": proposal["proposal_id"],
        "seed": args.seed,
        "stats": {"pool": split_stats(pool, len(rows)), TRAIN_ANNOTATED: split_stats(picked, len(rows))},
        "per_lab": per_lab,
        "episodes": {
            TRAIN_ANNOTATED: [
                {
                    "dataset_repo": r["dataset_repo"],
                    "repo_path": r["repo_path"],
                    "lab_id": r["lab_id"],
                    "robot_name": r["robot_name"],
                    "robot": r["robot"],
                    "task": r["task"],
                }
                for r in sorted(picked, key=key)
            ]
        },
    }
    args.out.write_text(json.dumps(out, indent=2))

    s = out["stats"][TRAIN_ANNOTATED]
    p = out["stats"]["pool"]
    counts = per_lab.values()
    print(
        f"{s['episodes']} episodes from {s['labs']}/{p['labs']} labs, {s['tasks']}/{p['tasks']} tasks, "
        f"{s['robots']}/{p['robots']} robots (per lab {min(counts)}-{max(counts)}, "
        f"success rate {s['success_rate']:.2f})",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
