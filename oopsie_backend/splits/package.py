"""Zip the three eval splits of one proposal for review.

Usage:
    uv run oopsie splits package \\
        --proposal /corral/utexas/IRI26024/oopsiedata/annotation/proposals/proposal_01.json \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir review_bundles

Writes ``id.zip``, ``od_task_id_scene.zip`` and ``od_task_od_scene.zip``. Each
unpacks to ``<split>/`` with ``eval_manifest.json`` and ``episodes/0001.h5``,
``episodes/0001_<cam>.mp4``, ... in the layout of the annotation bundles, so
``oopsie-data annotate --samples-dir <split>/episodes`` opens it. Episodes are
ordered by lab and task. The copies keep their annotations and source data,
and the manifest lists each episode's source, lab, robot and task.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from oopsie_backend.annotation.bundles import write_bundle
from oopsie_backend.common.repos import DEFAULT_ANNOTATION_DIR, DEFAULT_DATA_DIR
from oopsie_backend.splits.eval_split import SPLITS


def split_entries(proposal: dict, split: str) -> list[dict]:
    """Episodes of ``split`` ordered by lab, task and path, numbered from 1."""
    rows = sorted(proposal["episodes"][split], key=lambda r: (r["lab_id"], r["task"], r["dataset_repo"], r["repo_path"]))
    return [{**r, "split": split, "position": i} for i, r in enumerate(rows, 1)]


def _package(job: tuple) -> tuple[str, list[str]]:
    split, entries, data_dir, out_dir, proposal_id = job
    path, missing = write_bundle(split, entries, data_dir, out_dir, proposal_id, blind=False)
    return str(path), missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", type=Path, required=True, help="proposal_XX.json or eval_proposal.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ANNOTATION_DIR / "review_bundles")
    parser.add_argument("--only", nargs="+", choices=SPLITS, metavar="SPLIT", help="Package only these splits")
    args = parser.parse_args(argv)

    proposal = json.loads(args.proposal.read_text())
    proposal_id = proposal.get("proposal_id", args.proposal.stem)
    splits = args.only or list(SPLITS)
    out_dir = args.out_dir / proposal_id
    jobs = [(s, split_entries(proposal, s), args.data_dir, out_dir, proposal_id) for s in splits]

    failed = 0
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for path, missing in pool.map(_package, jobs):
            for msg in missing:
                print(f"  {Path(path).stem}: {msg}", file=sys.stderr)
            failed += len(missing)
    print(f"{len(jobs)} zip(s) in {out_dir}, {failed} episode(s) not packaged", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
