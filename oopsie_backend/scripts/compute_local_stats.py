"""Compute dashboard stats from synced local HF data on disk.

Usage:
    uv run python -m oopsie_backend.scripts.compute_local_stats \\
        --data-dir /scratch/09666/abahety/oopsiedata
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from oopsie_backend.utils.episode_stats import DEFAULT_STATS_JSON, aggregate, extract_record

DEFAULT_DATA_DIR = Path("/scratch/09666/abahety/oopsiedata")


def discover_episodes(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``(dataset_repo, repo_relative_path, absolute_path)`` for each .h5."""
    episodes: list[tuple[str, str, Path]] = []
    for path in sorted(data_dir.rglob("*.h5")):
        if ".cache" in path.parts:
            continue
        rel = path.relative_to(data_dir)
        if len(rel.parts) < 3:
            continue
        namespace, repo_name = rel.parts[0], rel.parts[1]
        repo_path = "/".join(rel.parts[2:])
        dataset_repo = f"{namespace}/{repo_name}"
        episodes.append((dataset_repo, repo_path, path))
    return episodes


def collect_local_episodes(
    episodes: list[tuple[str, str, Path]],
    file_limit: int | None,
) -> tuple[list[dict], int, list[str]]:
    """Read local .h5 files and return ``(records, skipped, datasets_included)``."""
    by_dataset: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for dataset_repo, repo_path, path in episodes:
        by_dataset[dataset_repo].append((repo_path, path))

    records: list[dict] = []
    skipped = 0
    datasets_included: list[str] = []

    for dataset_repo in sorted(by_dataset):
        files = by_dataset[dataset_repo]
        to_process = files if file_limit is None else files[:file_limit]
        datasets_included.append(dataset_repo)
        print(f"  dataset {dataset_repo}: {len(to_process)} .h5 file(s)", file=sys.stderr)

        for i, (repo_path, path) in enumerate(to_process, 1):
            try:
                record = extract_record(path, repo_path)
                record["dataset_repo"] = dataset_repo
                records.append(record)
            except Exception as exc:
                skipped += 1
                print(
                    f"  [{i}/{len(to_process)}] {dataset_repo} :: {repo_path}: "
                    f"SKIPPED ({exc})",
                    file=sys.stderr,
                )

    return records, skipped, datasets_included


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root of synced local data (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STATS_JSON,
        help="Where to write stats.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max .h5 files per dataset (first N paths in each dataset folder).",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    episodes = discover_episodes(data_dir)
    if not episodes:
        print(f"error: no .h5 files found under {data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(episodes)} .h5 file(s) under {data_dir}", file=sys.stderr)
    records, skipped, datasets_included = collect_local_episodes(episodes, args.limit)

    aggregate_label = f"local:{data_dir}"
    stats = aggregate(records, aggregate_label)
    stats["totals"]["read_errors"] = skipped
    stats["datasets_included"] = datasets_included

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    t = stats["totals"]
    print(
        f"\nWrote {args.output}\n  {t['episodes']} episodes, {t['labs']} labs, "
        f"{t['datasets']} datasets, {t['successes']}S / {t['failures']}F "
        f"over {t['duration_hours']}h",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
