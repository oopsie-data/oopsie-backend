"""Render a standalone HTML public report from public_stats.json.

Usage:
    uv run python -m oopsie_backend.scripts.render_public_report \\
        --input reports/public_stats.json \\
        --output reports/public_stats.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.utils.episode_stats import DEFAULT_PUBLIC_STATS_HTML, DEFAULT_PUBLIC_STATS_JSON
from oopsie_backend.utils.public_report import load_public_stats, render_public_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PUBLIC_STATS_JSON,
        help="Path to public_stats.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PUBLIC_STATS_HTML,
        help="HTML output path.",
    )
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    render_public_report(load_public_stats(input_path), args.output.resolve())
    print(f"Wrote public report: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
