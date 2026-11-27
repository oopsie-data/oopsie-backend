"""Record the chosen eval split proposal in the episode index.

Usage:
    uv run python -m oopsie_backend.splits.record \\
        --proposal eval_split_proposals/proposal_02.json --index episode_index.parquet

Sets ``eval_split`` and ``eval_proposal`` for every episode the proposal holds
out or excludes and for every other episode of its held-out labs and tasks, and
clears them everywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.index.store import DEFAULT_INDEX, known_only, read_index, update_columns
from oopsie_backend.splits.columns import EVAL_COLUMNS, eval_split_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(argv)

    proposal = json.loads(args.proposal.read_text())
    index = read_index(args.index, columns=["lab_id", "task"])
    frame = known_only(eval_split_frame(proposal, index), args.index, "proposal")
    update_columns(args.index, frame, EVAL_COLUMNS, clear_missing=True)
    counts = frame.group_by("eval_split").len().sort("eval_split").iter_rows()
    print(f"recorded {', '.join(f'{s} {n}' for s, n in counts)} in {args.index}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
