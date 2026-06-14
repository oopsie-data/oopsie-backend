"""Stream every .h5 in a private HuggingFace dataset into memory, extract
metadata, and write a single aggregate `stats.json` consumed by the Jekyll
dashboard at `oopsie-docs/docs/stats.md`.

No file is ever persisted to disk except the final JSON.

Usage:
    export HF_TOKEN=hf_...
    python sweep_dataset.py --repo namespace/dataset-name \\
        --output ../oopsie-docs/docs/assets/data/stats.json

    # Or paste the dataset page URL:
    python sweep_dataset.py --repo https://huggingface.co/datasets/namespace/dataset-name

    # Sweep every dataset repo under a Hugging Face organization (same token):
    python sweep_dataset.py --org OopsieData-Submissions
    python sweep_dataset.py --org https://huggingface.co/OopsieData-Submissions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError

from oopsie_backend.utils.episode_stats import DEFAULT_STATS_JSON, aggregate, extract_record
from oopsie_backend.utils.hf import (
    list_dataset_repos_for_org,
    list_h5_files,
    parse_org_arg,
    parse_repo_arg,
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
        "--output",
        type=Path,
        default=DEFAULT_STATS_JSON,
        help="Where to write stats.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max .h5 files to attempt per dataset (first N paths in each repo).",
    )
    args = parser.parse_args(argv)

    records: list[dict] = []
    skipped = 0
    datasets_included: list[str] = []
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
    stats = aggregate(records, aggregate_label)
    stats["totals"]["read_errors"] = skipped
    if datasets_included:
        stats["datasets_included"] = datasets_included
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    t = stats["totals"]
    print(
        f"\nWrote {args.output}\n  {t['episodes']} episodes, {t['labs']} labs, "
        f"{t['successes']}S / {t['failures']}F over {t['duration_hours']}h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
