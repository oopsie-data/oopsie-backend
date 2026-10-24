"""Write per-lab feedback text from action_qa.json: an "Automated checks" block per lab,
on its own (lab_feedback_automated.txt) and appended to each lab's section of the
hand-written visual-review drafts (lab_feedback_combined.txt). The drafts file itself is
never modified. Which checks appear, and their confidence/action tags, come from the
``confidence:`` / ``action:`` fields in configs/action_qa_config.yaml.

Usage:
    uv run python -m oopsie_backend.qa.render_lab_feedback \\
        --input reports/action_qa.json \\
        --drafts reports/lab_feedback_drafts.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oopsie_backend.qa.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.qa.lab_feedback import (
    automated_report,
    render_automated,
    render_combined,
)


def main(argv: list[str] | None = None) -> int:
    reports_dir = ACTION_QA_CONFIG["run"]["output"].parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=ACTION_QA_CONFIG["run"]["output"],
        help="Path to action_qa.json (default: run.output from action_qa_config.yaml).",
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=reports_dir / "lab_feedback_drafts.txt",
        help="Visual-review drafts to merge into (default: lab_feedback_drafts.txt next to the report).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=reports_dir,
        help="Where to write lab_feedback_automated.txt and lab_feedback_combined.txt.",
    )
    parser.add_argument("--examples", type=int, default=3, help="Example episodes per item (default: 3).")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1
    report = json.loads(args.input.read_text(encoding="utf-8"))
    blocks = automated_report(report, ACTION_QA_CONFIG["lab_feedback"], args.examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    automated_path = args.output_dir / "lab_feedback_automated.txt"
    automated_path.write_text(render_automated(blocks), encoding="utf-8")
    print(f"Wrote {automated_path} ({len(blocks)} labs)", file=sys.stderr)

    if args.drafts.is_file():
        combined, unmatched = render_combined(args.drafts.read_text(encoding="utf-8"), blocks)
        combined_path = args.output_dir / "lab_feedback_combined.txt"
        combined_path.write_text(combined, encoding="utf-8")
        print(f"Wrote {combined_path}", file=sys.stderr)
        if unmatched:
            print(
                f"note: no QA results for draft section(s) {unmatched} - add them to "
                "LAB_NAME_ALIASES in oopsie_backend/qa/lab_feedback.py if the name differs",
                file=sys.stderr,
            )
    else:
        print(f"note: drafts not found ({args.drafts}); wrote the automated file only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
