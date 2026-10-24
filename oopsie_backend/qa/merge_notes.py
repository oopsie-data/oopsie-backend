"""Collect QA notes from returned ``qa_manifest.json`` files into one CSV.

Usage:
    uv run python -m oopsie_backend.qa.merge_notes \\
        returned/*.json --out qa_notes.csv

Each row identifies the episode by ``dataset_repo`` and ``repo_path`` (the h5 path
inside that HF dataset repo). ``--index`` also writes the notes to the episode
index as ``qa_notes`` (one entry per returned note), for the episodes in these
manifests only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from oopsie_backend.index.store import KEY, episode_key, known_only, update_columns

QA_NOTE = pl.Struct(
    {"bundle_id": pl.String, "qa_id": pl.String, "reviewer": pl.String, "updated_at": pl.String, "notes": pl.String}
)

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
    parser.add_argument("--index", type=Path, help="Also write the notes to this episode index")
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

    if args.index is not None:
        notes: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if str(r["qa_notes"] or "").strip():
                notes[episode_key(r["dataset_repo"], r["repo_path"])].append(
                    {
                        "bundle_id": r["bundle_id"],
                        "qa_id": r["qa_id"],
                        "reviewer": r["reviewer"],
                        "updated_at": r["qa_updated_at"] or None,
                        "notes": r["qa_notes"],
                    }
                )
        frame = pl.DataFrame(
            [{KEY: k, "qa_notes": v} for k, v in notes.items()],
            schema={KEY: pl.String, "qa_notes": pl.List(QA_NOTE)},
        )
        frame = known_only(frame, args.index, "noted")
        update_columns(args.index, frame, ["qa_notes"])
        print(f"wrote notes for {frame.height} episode(s) to {args.index}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
