"""Combine existing per-detector benchmark summaries without rerunning models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oopsie_backend.face_blur.benchmark import compare_existing_reports
from oopsie_backend.face_blur.models import MODEL_SPECS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("face_blur_benchmarks/latest")
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=None,
    )
    args = parser.parse_args(argv)
    try:
        compare_existing_reports(
            args.output_dir.resolve(), args.detectors or MODEL_SPECS
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison: {args.output_dir.resolve() / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
