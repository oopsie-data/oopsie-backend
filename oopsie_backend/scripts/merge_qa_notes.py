"""Collect QA notes from returned ``qa_manifest.json`` files into one CSV.

Usage:
    uv run python -m oopsie_backend.scripts.merge_qa_notes \\
        returned/*.json --out qa_notes.csv

Each row identifies the episode by ``dataset_repo`` and ``repo_path`` (the h5 path
inside that HF dataset repo).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = (
    "bundle_id",
    "qa_id",
    "reviewer",
    "dataset_repo",
    "repo_path",
    "lab_id",
    "operator",
    "annotators",
    "robot_name",
    "language_instruction",
    "qa_updated_at",
    "qa_notes",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("qa_notes.csv"))
    parser.add_argument(
        "--all", action="store_true", help="Include episodes without QA notes."
    )
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for path in args.manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest.get("episodes", []):
            if not args.all and not str(entry.get("qa_notes") or "").strip():
                continue
            rows.append(
                {
                    **{k: entry.get(k, "") for k in COLUMNS},
                    "bundle_id": manifest.get("bundle_id", ""),
                    "reviewer": entry.get("qa_reviewer") or manifest.get("reviewer", ""),
                    "annotators": ";".join(entry.get("annotators") or []),
                }
            )

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
