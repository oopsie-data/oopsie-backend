"""Select diverse episodes per lab for QA spot checks and write zip bundles.

Usage:
    uv run python -m oopsie_backend.qa.select_spot_checks \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir qa_out

Each lab (HF dataset repo) contributes up to ``--per-lab`` episodes chosen to
cover as many language instructions, annotators, operators and robots as
possible; labs with fewer than ``--min-episodes`` eligible episodes are
skipped. Labs are grouped ``--labs-per-bundle`` per zip; every zip holds the
h5 and mp4 files plus ``qa_manifest.json`` with an empty ``qa_notes`` field
per episode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.common.repos import DEFAULT_DATA_DIR, discover_episodes
from oopsie_backend.qa.spot_checks import (
    collect_records,
    diversity_report,
    select_per_lab,
    write_bundles,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("qa_out"))
    parser.add_argument("--per-lab", type=int, default=20)
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=200,
        help="Skip labs with fewer eligible episodes (use 0 for test data).",
    )
    parser.add_argument("--labs-per-bundle", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-lab diversity table without writing bundles.",
    )
    args = parser.parse_args(argv)

    episodes = discover_episodes(args.data_dir)
    if not episodes:
        print(f"error: no .h5 files under {args.data_dir}", file=sys.stderr)
        return 1
    records = collect_records(episodes, args.data_dir, workers=args.workers)
    selection = select_per_lab(records, args.per_lab, args.seed, args.min_episodes)

    print("distinct values selected/available per dimension:")
    for line in diversity_report(records, selection):
        print(line)

    if not args.dry_run:
        write_bundles(selection, args.out_dir, args.labs_per_bundle, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
