"""Stream every .h5 in a private HuggingFace dataset into memory, extract
metadata, and write public + internal stats artifacts.

Usage:
    export HF_TOKEN=hf_...
    uv run python -m oopsie_backend.stats.compute_hf \\
        --repo OopsieData-Submissions/SomeLab

    uv run python -m oopsie_backend.stats.compute_hf \\
        --org OopsieData-Submissions
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError

from oopsie_backend.common.hf import (
    list_dataset_created_dates_for_org,
    list_dataset_repos_for_org,
    list_h5_files,
    parse_org_arg,
    parse_repo_arg,
)
from oopsie_backend.common.repos import (
    PUBLIC_EXCLUDED_DATASETS,
    STATS_EXCLUDED_DATASETS,
)
from oopsie_backend.stats.episode_stats import (
    DEFAULT_DOCS_STATS_JSON,
    DEFAULT_INTERNAL_STATS_JSON,
    DEFAULT_PUBLIC_STATS_HTML,
    DEFAULT_PUBLIC_STATS_JSON,
    aggregate,
    extract_record,
    public_dataset_repos,
    write_internal_artifacts,
    write_public_artifacts,
)


def fetch_episode_bytes(
    repo: str, path: str, token: str | None, repo_type: str
) -> bytes:
    """Stream one .h5 into memory via an authenticated HTTPS GET."""
    url = hf_hub_url(repo_id=repo, filename=path, repo_type=repo_type)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    return r.content


def collect_episodes(
    repo_id: str,
    repo_type: str,
    token: str | None,
    files: list[str],
    file_limit: int | None,
) -> tuple[list[dict], int]:
    """Read each listed ``.h5`` into memory and return ``(records, skipped)``."""
    to_process = files if file_limit is None else files[:file_limit]
    records: list[dict] = []
    skipped = 0
    n = len(to_process)
    for i, path in enumerate(to_process, 1):
        try:
            buf = fetch_episode_bytes(repo_id, path, token, repo_type)
            record = extract_record(buf, path)
            record["dataset_repo"] = repo_id
            records.append(record)
        except Exception as exc:
            skipped += 1
            print(
                f"  [{i}/{n}] {repo_id} :: {path}: SKIPPED ({exc})",
                file=sys.stderr,
            )
            continue
    return records, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--repo",
        help=(
            "Single HuggingFace dataset id `namespace/name`, or a full URL "
            "`https://huggingface.co/datasets/namespace/name`."
        ),
    )
    src.add_argument(
        "--org",
        help=(
            "Hugging Face organization slug (or org profile URL). "
            "Every dataset under that org (visible to your token) is swept."
        ),
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="Hub repo type (default: dataset). Use `model` only if .h5 files live in a model repo.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF read token (defaults to $HF_TOKEN).",
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
        help="Max .h5 files to attempt per dataset (first N paths in each repo).",
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

    records: list[dict] = []
    skipped = 0
    datasets_included: list[str] = []
    dataset_created_at: dict[str, str | None] = {}
    aggregate_label: str

    if args.repo is not None:
        try:
            repo_id, repo_type = parse_repo_arg(args.repo, args.repo_type)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        try:
            files = list_h5_files(repo_id, args.token, repo_type)
        except RepositoryNotFoundError:
            print(
                f"error: repository not found: {repo_id!r} (repo_type={repo_type!r}).\n"
                "  - Check `namespace/repo` (dataset URLs contain `/datasets/`).\n"
                "  - For private repos, ensure HF_TOKEN has read access.",
                file=sys.stderr,
            )
            return 1
        print(f"Found {len(files)} .h5 file(s) in {repo_id} ({repo_type})", file=sys.stderr)
        batch, sk = collect_episodes(
            repo_id, repo_type, args.token, files, args.limit
        )
        records.extend(batch)
        skipped += sk
        aggregate_label = repo_id

    else:
        assert args.org is not None
        try:
            org_slug = parse_org_arg(args.org)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        try:
            dataset_ids = list_dataset_repos_for_org(org_slug, args.token)
        except Exception as exc:
            print(f"error: could not list datasets for org {org_slug!r}: {exc}", file=sys.stderr)
            return 1

        try:
            dataset_created_at = list_dataset_created_dates_for_org(org_slug, args.token)
        except Exception as exc:
            print(
                f"  (could not fetch dataset creation dates: {exc}; "
                "internal report will fall back to episode timestamps)",
                file=sys.stderr,
            )
            dataset_created_at = {}

        if not dataset_ids:
            print(
                f"error: no datasets returned for author={org_slug!r}. "
                "Check the org slug and HF_TOKEN read access.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Found {len(dataset_ids)} dataset repo(s) under org {org_slug!r}",
            file=sys.stderr,
        )
        aggregate_label = f"{org_slug}/*"

        for ds_id in dataset_ids:
            if ds_id in STATS_EXCLUDED_DATASETS:
                print(
                    f"  skip (STATS_EXCLUDED_DATASETS): {ds_id}",
                    file=sys.stderr,
                )
                continue
            try:
                files = list_h5_files(ds_id, args.token, "dataset")
            except RepositoryNotFoundError:
                print(f"  skip (not found): {ds_id}", file=sys.stderr)
                continue
            if not files:
                print(f"  skip (no .h5): {ds_id}", file=sys.stderr)
                continue

            print(f"  dataset {ds_id}: {len(files)} .h5 file(s)", file=sys.stderr)
            datasets_included.append(ds_id)
            batch, sk = collect_episodes(
                ds_id, "dataset", args.token, files, args.limit
            )
            records.extend(batch)
            skipped += sk
    stats = aggregate(records, aggregate_label, include_excluded=False)
    stats["totals"]["read_errors"] = skipped
    if datasets_included:
        stats["datasets_included"] = public_dataset_repos(datasets_included)
    elif (
        args.repo is not None
        and aggregate_label not in PUBLIC_EXCLUDED_DATASETS
        and aggregate_label not in STATS_EXCLUDED_DATASETS
    ):
        stats["datasets_included"] = [aggregate_label]

    write_public_artifacts(
        stats,
        backend_json_path=args.public_json,
        docs_json_path=args.docs_output,
        html_path=args.public_html,
        write_html=not args.no_public_report,
    )
    if not args.no_internal_output:
        write_internal_artifacts(
            records,
            aggregate_label,
            args.internal_output,
            write_html=not args.no_internal_report,
            all_dataset_repos=list(dataset_created_at) if dataset_created_at else None,
            dataset_created_at=dataset_created_at or None,
        )

    t = stats["totals"]
    print(
        f"\nPublic stats JSON: {args.public_json}\n"
        f"  Docs site JSON: {args.docs_output}\n"
        f"  {t['episodes']} episodes, {t['labs']} labs, "
        f"{t['successes']}S / {t['failures']}F over {t['duration_hours']}h"
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
