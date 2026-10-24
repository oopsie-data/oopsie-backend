"""Collect annotations from returned eval bundles.

Usage:
    uv run python -m oopsie_backend.scripts.collect_eval_annotations \\
        returned/*.zip returned/annotator_07/ \\
        --assignment-dir eval_assignment --store eval_annotations.jsonl

Inputs are returned zips, unpacked bundle folders or single h5 files. Each
file is matched to its source episode by its blinded ``episode_id``. New
annotations are appended to ``--store``; nothing in it is overwritten, so the
command can be re-run as returns arrive. A coverage summary is written next
to the store.

``--apply-to DIR`` (with ``--data-dir``) additionally writes copies of the
source episodes holding every collected annotation as its own group.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.scripts.compute_local_stats import DEFAULT_DATA_DIR
from oopsie_backend.utils.eval_collect import (
    apply_to_copies,
    coverage_report,
    load_assignment,
    merge_into_store,
    read_returns,
    read_store,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("returns", nargs="*", type=Path, help="Returned zips, folders or h5 files")
    parser.add_argument("--assignment-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("eval_annotations.jsonl"))
    parser.add_argument(
        "--bundle", help="Bundle (annotator list) name for returns without an eval_manifest.json"
    )
    parser.add_argument("--apply-to", type=Path, metavar="DIR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    assignment = load_assignment(args.assignment_dir)
    if not assignment:
        print(f"error: no annotator lists in {args.assignment_dir}", file=sys.stderr)
        return 1
    records, problems = read_returns(args.returns, assignment, args.bundle)
    for msg in problems:
        print(f"  skipped {msg}", file=sys.stderr)
    added, present = merge_into_store(args.store, records)
    print(f"{len(records)} annotation(s) read: {added} added, {present} already in {args.store}", file=sys.stderr)

    store = read_store(args.store)
    report = coverage_report(store, assignment)
    summary_path = args.store.with_name(args.store.stem + "_summary.json")
    summary_path.write_text(json.dumps(report, indent=2))
    print(
        f"returned {report['annotations_returned']}/{report['annotations_assigned']} assigned annotations, "
        f"{report['revised_annotations']} revised, {report['unexpected_returns']} unexpected; see {summary_path}",
        file=sys.stderr,
    )

    if args.apply_to:
        apply_to_copies(store, assignment, args.data_dir, args.apply_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
