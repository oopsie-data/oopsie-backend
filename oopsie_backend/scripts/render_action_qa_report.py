"""Render a standalone HTML page from action_qa.json (run_action_qa.py also does this
automatically after each run; use this to re-render without re-running the checks).

Usage:
    uv run python -m oopsie_backend.scripts.render_action_qa_report \\
        --input reports/action_qa.json \\
        --output reports/action_qa.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.utils.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.utils.action_qa_report import render_action_qa_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ACTION_QA_CONFIG["run"]["output"],
        help="Path to action_qa.json (default: run.output from action_qa_config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: same stem as --input with .html).",
    )
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_suffix(".html")
    report = json.loads(input_path.read_text(encoding="utf-8"))
    render_action_qa_report(report, output_path, ACTION_QA_CONFIG["report"]["highlight_ignore"])
    print(f"Wrote action QA report: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
