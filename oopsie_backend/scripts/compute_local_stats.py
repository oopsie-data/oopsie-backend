"""Compute dashboard stats from synced local HF data on disk.

Usage:
    uv run python -m oopsie_backend.scripts.compute_local_stats \\
        --data-dir /scratch/09666/abahety/oopsiedata

Writes by default:
  - reports/public_stats.json + reports/public_stats.html (backend)
  - ../oopsie-docs/docs/assets/data/stats.json (public docs site)
  - reports/internal_stats.json + reports/internal_stats.html
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from oopsie_backend.utils.episode_stats import (
    DEFAULT_DOCS_STATS_JSON,
    DEFAULT_INTERNAL_STATS_JSON,
    DEFAULT_PUBLIC_STATS_HTML,
    DEFAULT_PUBLIC_STATS_JSON,
    HF_ORG,
    STATS_EXCLUDED_DATASETS,
    aggregate,
    extract_record,
    public_dataset_repos,
    resolve_local_episode_path,
    write_internal_artifacts,
    write_public_artifacts,
)

DEFAULT_DATA_DIR = Path("/scratch/09666/abahety/oopsiedata")


def discover_all_dataset_dirs(data_dir: Path) -> list[str]:
    """Return every dataset repo id present on disk, including ones with zero .h5
    files (e.g. a lab registered but hasn't submitted episodes yet)."""
    data_dir = data_dir.resolve()
    root = data_dir if data_dir.name == HF_ORG else data_dir / HF_ORG
    if not root.is_dir():
        return []
    return sorted(f"{HF_ORG}/{p.name}" for p in root.iterdir() if p.is_dir())


def fetch_dataset_created_dates(org: str) -> dict[str, str | None]:
    """Best-effort fetch of true HF repo creation dates for the org's dataset repos.

    Returns ``{}`` (not an error) when there's no usable token or the Hub can't be
    reached; the internal report then falls back to episode-timestamp heuristics.
    """
    try:
        from huggingface_hub import get_token

        from oopsie_backend.utils.hf import list_dataset_created_dates_for_org

        token = get_token()
        if not token:
            print(
                "  (no cached HF token; dataset 'created' dates will fall back to "
                "episode timestamps)",
                file=sys.stderr,
            )
            return {}
        dates = list_dataset_created_dates_for_org(org, token)
        print(
            f"  Fetched creation dates for {len(dates)} dataset repo(s) from the Hub",
            file=sys.stderr,
        )
        return dates
    except Exception as exc:
        print(
            f"  (could not fetch dataset creation dates from the Hub: {exc})",
            file=sys.stderr,
        )
        return {}


def discover_episodes(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``(dataset_repo, repo_relative_path, absolute_path)`` for each .h5."""
    data_dir = data_dir.resolve()
    episodes: list[tuple[str, str, Path]] = []
    for path in sorted(data_dir.rglob("*.h5")):
        if ".cache" in path.parts:
            continue
        rel = path.relative_to(data_dir)
        resolved = resolve_local_episode_path(data_dir, rel)
        if resolved is None:
            continue
        dataset_repo, repo_path = resolved
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
        if dataset_repo in STATS_EXCLUDED_DATASETS:
            n = len(by_dataset[dataset_repo])
            print(
                f"  dataset {dataset_repo}: SKIPPED ({n} .h5 file(s); "
                f"in STATS_EXCLUDED_DATASETS)",
                file=sys.stderr,
            )
            continue
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
        "--docs-output",
        type=Path,
        default=DEFAULT_DOCS_STATS_JSON,
        help="Public stats JSON for the docs site (stats.json).",
    )
    parser.add_argument(
        "--public-json",
        type=Path,
        default=DEFAULT_PUBLIC_STATS_JSON,
        help="Public stats JSON copy under oopsie-backend/reports/.",
    )
    parser.add_argument(
        "--public-html",
        type=Path,
        default=DEFAULT_PUBLIC_STATS_HTML,
        help="Public stats HTML snapshot under oopsie-backend/reports/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max .h5 files per dataset (first N paths in each dataset folder).",
    )
    parser.add_argument(
        "--no-public-report",
        action="store_true",
        help="Skip public_stats.html (JSON copies are still written).",
    )
    parser.add_argument(
        "--internal-output",
        type=Path,
        default=DEFAULT_INTERNAL_STATS_JSON,
        help="Internal stats JSON under oopsie-backend/reports/.",
    )
    parser.add_argument(
        "--no-internal-output",
        action="store_true",
        help="Skip internal_stats.json and internal report.",
    )
    parser.add_argument(
        "--no-internal-report",
        action="store_true",
        help="Skip internal_stats.html when internal JSON is written.",
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
    stats = aggregate(records, aggregate_label, include_excluded=False)
    stats["totals"]["read_errors"] = skipped
    stats["datasets_included"] = public_dataset_repos(datasets_included)

    write_public_artifacts(
        stats,
        backend_json_path=args.public_json,
        docs_json_path=args.docs_output,
        html_path=args.public_html,
        write_html=not args.no_public_report,
    )
    if not args.no_internal_output:
        all_dataset_repos = discover_all_dataset_dirs(data_dir)
        dataset_created_at = fetch_dataset_created_dates(HF_ORG)
        # The Hub may know about repos not yet synced to local disk; include those too.
        all_dataset_repos = sorted(set(all_dataset_repos) | set(dataset_created_at))
        write_internal_artifacts(
            records,
            aggregate_label,
            args.internal_output,
            write_html=not args.no_internal_report,
            all_dataset_repos=all_dataset_repos,
            dataset_created_at=dataset_created_at,
        )

    t = stats["totals"]
    print(
        f"\nPublic stats JSON: {args.public_json}\n"
        f"  Docs site JSON: {args.docs_output}\n"
        f"  {t['episodes']} episodes, {t['labs']} labs, {t['datasets']} datasets, "
        f"{t['successes']}S / {t['failures']}F over {t['duration_hours']}h",
        file=sys.stderr,
    )
    if not args.no_public_report:
        print(f"  Public report: {args.public_html}", file=sys.stderr)
    if not args.no_internal_output:
        print(f"  Internal stats: {args.internal_output}", file=sys.stderr)
        if not args.no_internal_report:
            print(
                f"  Internal report: {args.internal_output.with_suffix('.html')}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
