"""Write a JSONL metadata index of every publicly included episode.

Usage:
    uv run python -m oopsie_backend.scripts.build_episode_index \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --out episode_index.jsonl

One line per episode with lab, robot, cameras, instruction, policy, outcome
and annotators. Videos are not read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.scripts.compute_local_stats import (
    DEFAULT_DATA_DIR,
    discover_episodes,
)
from oopsie_backend.utils.episode_index import build_index, write_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("episode_index.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    episodes = discover_episodes(args.data_dir)
    if not episodes:
        print(f"error: no .h5 files under {args.data_dir}", file=sys.stderr)
        return 1
    rows = build_index(episodes, workers=args.workers)
    write_index(rows, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
