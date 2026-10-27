"""Render the QA report page from the index's ``qa_*`` columns without running the checks
(``qa run`` also writes it after each run).

Usage:
    uv run python -m oopsie_backend.qa.render_report \\
        --index episode_index.parquet \\
        --output reports/action_qa.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.index.store import DEFAULT_INDEX
from oopsie_backend.qa.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.qa.action_qa_report import render_action_qa_report
from oopsie_backend.qa.report import index_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--output",
        type=Path,
        default=ACTION_QA_CONFIG["run"]["output"],
        help="HTML output path (default: run.output from action_qa_config.yaml).",
    )
    args = parser.parse_args(argv)

    if not args.index.is_file():
        print(f"error: index not found: {args.index}", file=sys.stderr)
        return 1
    report = index_report(args.index, ACTION_QA_CONFIG)
    render_action_qa_report(report, args.output, ACTION_QA_CONFIG["report"]["highlight_ignore"])
    print(f"Wrote action QA report: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
