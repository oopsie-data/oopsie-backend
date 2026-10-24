"""Compute dashboard stats from the episode index.

Usage:
    uv run python -m oopsie_backend.stats.compute_local \\
        --index episode_index.parquet --data-dir /corral/utexas/IRI26024/oopsiedata

Counts readable, on-disk episodes that are neither a duplicate copy nor critical. ``--data-dir`` is only
used to list every dataset repo on disk, including ones without episodes, for the
internal report.

Writes by default:
  - reports/public_stats.json + reports/public_stats.html (backend)
  - ../oopsie-docs/docs/assets/data/stats.json (public docs site)
  - reports/internal_stats.json + reports/internal_stats.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from oopsie_backend.common.repos import DEFAULT_DATA_DIR, HF_ORG, discover_all_dataset_dirs
from oopsie_backend.index.store import DEFAULT_INDEX, read_index
from oopsie_backend.stats.episode_stats import (
    DEFAULT_DOCS_STATS_JSON,
    DEFAULT_INTERNAL_STATS_JSON,
    DEFAULT_PUBLIC_STATS_HTML,
    DEFAULT_PUBLIC_STATS_JSON,
    aggregate,
    public_dataset_repos,
    write_internal_artifacts,
    write_public_artifacts,
)
from oopsie_backend.stats.records import critical_summary, records_from_index


def fetch_dataset_created_dates(org: str) -> dict[str, str | None]:
    """Best-effort fetch of true HF repo creation dates for the org's dataset repos.

    Returns ``{}`` (not an error) when there's no usable token or the Hub can't be
    reached; the internal report then falls back to episode-timestamp heuristics.
    """
    try:
        from huggingface_hub import get_token

        from oopsie_backend.common.hf import list_dataset_created_dates_for_org

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
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Episode index to read")
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

    df = read_index(
        args.index,
        optional={
            "duplicate_of": pl.String,
            "critical": pl.Boolean,
            "critical_reasons": pl.List(pl.String),
            "qa_checked": pl.List(pl.String),
        },
    )
    records = records_from_index(df)
    on_disk = df.filter(pl.col("on_disk"))
    read_errors = on_disk.filter(pl.col("read_error").is_not_null()).height
    duplicates = on_disk.filter(pl.col("duplicate_of").is_not_null()).height
    critical = on_disk.filter(pl.col("critical").fill_null(False)).height
    print(
        f"{len(records)} episode(s) from {args.index} ({duplicates} duplicate copies, {critical} critical "
        f"and {read_errors} unreadable file(s) left out)",
        file=sys.stderr,
    )

    aggregate_label = f"local:{data_dir}"
    stats = aggregate(records, aggregate_label, include_excluded=False)
    stats["totals"]["read_errors"] = read_errors
    stats["datasets_included"] = public_dataset_repos(sorted({r["dataset_repo"] for r in records}))

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
            critical=critical_summary(df),
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
