"""Zip each annotator's eval list with its h5 and mp4 files.

Usage:
    uv run python -m oopsie_backend.annotation.package \\
        --assignment-dir eval_assignment \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir eval_bundles

Reads the ``annotator_XX.json`` files from ``assign_eval_annotators`` and
writes one ``<annotator>.zip`` each, unpacking to ``<annotator>/`` with
``eval_manifest.json`` and ``episodes/0001.h5``, ``episodes/0001_<cam>.mp4``, ...
Annotators run ``oopsie-data annotate --samples-dir <annotator>/episodes``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from oopsie_backend.annotation.bundles import write_bundle
from oopsie_backend.common.repos import DEFAULT_DATA_DIR, annotation_dir


def _package(job: tuple) -> tuple[str, list[str]]:
    path, data_dir, out_dir, blind = job
    lists = json.loads(Path(path).read_text())
    zip_path, missing = write_bundle(
        lists["annotator"], lists["episodes"], data_dir, out_dir, lists.get("proposal"), blind
    )
    return str(zip_path), missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, help="Default: <data-dir>/annotation/eval_bundles")
    parser.add_argument(
        "--keep-annotations",
        action="store_true",
        help="Keep existing annotations, source data and episode ids in the copies",
    )
    parser.add_argument("--only", nargs="+", metavar="ANNOTATOR", help="Package only these annotators")
    parser.add_argument("--workers", type=int, default=4, help="Bundles written in parallel")
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or annotation_dir(args.data_dir) / "eval_bundles"

    lists = sorted(args.assignment_dir.glob("*.json"))
    lists = [p for p in lists if p.name != "assignment_summary.json"]
    if args.only:
        lists = [p for p in lists if p.stem in set(args.only)]
    if not lists:
        print(f"error: no annotator lists in {args.assignment_dir}", file=sys.stderr)
        return 1

    jobs = [(p, args.data_dir, args.out_dir, not args.keep_annotations) for p in lists]
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for zip_path, missing in pool.map(_package, jobs):
            for msg in missing:
                print(f"  {Path(zip_path).stem}: {msg}", file=sys.stderr)
            failed += len(missing)
    print(f"{len(jobs)} bundle(s), {failed} episode(s) not packaged", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
