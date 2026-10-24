"""Render a standalone HTML internal report from internal_stats.json.

Usage:
    uv run python -m oopsie_backend.stats.render_internal \\
        --input reports/internal_stats.json \\
        --output reports/internal_stats.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.stats.episode_stats import DEFAULT_INTERNAL_STATS_JSON
from oopsie_backend.stats.internal_report import (
    load_internal_stats,
    render_internal_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INTERNAL_STATS_JSON,
        help="Path to internal_stats.json.",
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
    data = load_internal_stats(input_path)
    render_internal_report(data, output_path)
    print(f"Wrote internal report: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
