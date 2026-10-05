"""Sync every dataset under a HuggingFace org to a local directory.

Downloads only missing or changed files (ETag-based). Local files removed from
the Hub are kept on disk (local is a superset, not a strict mirror).

Usage:
    export HF_TOKEN=hf_...
    uv run python -m oopsie_backend.scripts.sync_hf_org \\
        --org OopsieData-Submissions \\
        --local-dir /scratch/09666/abahety/oopsiedata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.file_download import DryRunFileInfo
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

from oopsie_backend.utils.episode_stats import now_timestamps
from oopsie_backend.utils.hf import (
    list_dataset_repos_for_org,
    parse_org_arg,
    split_repo_id,
)

_INDENT = "  "


def _log(message: str, *, level: int = 0) -> None:
    print(f"{_INDENT * level}{message}", file=sys.stderr)


@contextmanager
def _suppress_hf_progress():
    """Hide huggingface_hub tqdm bars (snapshot_download uses these by default)."""
    disable_progress_bars()
    try:
        yield
    finally:
        enable_progress_bars()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


@dataclass
class DatasetSyncResult:
    repo_id: str
    duration_seconds: float
    files_downloaded: int = 0
    files_skipped: int = 0


@dataclass
class SyncStats:
    datasets_synced: int = 0
    files_downloaded: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    dataset_results: list[DatasetSyncResult] = field(default_factory=list)


def local_dataset_dir(local_dir: Path, repo_id: str) -> Path:
    namespace, repo_name = split_repo_id(repo_id)
    return local_dir / namespace / repo_name


def local_file_state(dataset_dir: Path) -> dict[str, tuple[int, int]]:
    """Return ``{repo_relative_path: (size_bytes, mtime_ns)}`` for synced files."""
    if not dataset_dir.is_dir():
        return {}
    state: dict[str, tuple[int, int]] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or ".cache" in path.parts:
            continue
        rel = path.relative_to(dataset_dir).as_posix()
        stat = path.stat()
        state[rel] = (stat.st_size, stat.st_mtime_ns)
    return state


def diff_file_state(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """Return ``(downloaded_or_updated, skipped_unchanged)``."""
    downloaded = 0
    skipped = 0
    for rel, after_stat in after.items():
        if before.get(rel) == after_stat:
            skipped += 1
        else:
            downloaded += 1
    return downloaded, skipped


def sync_dataset(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    *,
    dry_run: bool,
    verbose: bool,
) -> tuple[int, int]:
    """Sync one dataset repo. Returns ``(would_download_or_downloaded, skipped)``."""
    dest = local_dataset_dir(local_dir, repo_id)
    before = local_file_state(dest) if not dry_run else {}

    with _suppress_hf_progress():
        result = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest,
            token=token,
            dry_run=dry_run,
            # Default is 8; large video files sharing one connection at that
            # concurrency easily exceed the 10s per-chunk read timeout, which
            # aborts the whole sync after retries are exhausted. Lower contention
            # instead of just raising the timeout.
            max_workers=3,
        )

    if dry_run:
        infos: list[DryRunFileInfo] = result  # type: ignore[assignment]
        downloaded = sum(1 for info in infos if info.will_download)
        skipped = sum(1 for info in infos if not info.will_download)
        if verbose:
            for info in infos:
                if info.will_download:
                    _log(f"would download {info.filename}", level=2)
        return downloaded, skipped

    return diff_file_state(before, local_file_state(dest))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--org",
        required=True,
        help="Hugging Face organization slug or profile URL.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Root directory for synced local data.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF read token (defaults to $HF_TOKEN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be downloaded without writing files.",
    )
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=None,
        help="Max datasets to sync (for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-dataset and per-file actions.",
    )
    args = parser.parse_args(argv)

    org_slug = parse_org_arg(args.org)
    dataset_ids = list_dataset_repos_for_org(org_slug, args.token)
    if args.limit_datasets is not None:
        dataset_ids = dataset_ids[: args.limit_datasets]

    if not dataset_ids:
        print(
            f"error: no datasets returned for author={org_slug!r}. "
            "Check the org slug and HF_TOKEN read access.",
            file=sys.stderr,
        )
        return 1

    _log(
        f"{'Dry-run' if args.dry_run else 'Syncing'} {len(dataset_ids)} "
        f"dataset(s) under {org_slug!r}",
    )

    stats = SyncStats()
    total = len(dataset_ids)

    for index, repo_id in enumerate(dataset_ids, 1):
        label = f"[{index}/{total}] {repo_id}"
        dest = local_dataset_dir(args.local_dir, repo_id)
        if args.verbose:
            _log(label)
            _log(f"dest: {dest}", level=1)

        started = time.perf_counter()
        try:
            downloaded, skipped = sync_dataset(
                repo_id,
                args.local_dir,
                args.token,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        except RepositoryNotFoundError:
            elapsed = time.perf_counter() - started
            msg = f"skip (not found): {repo_id}"
            stats.dataset_results.append(
                DatasetSyncResult(repo_id=repo_id, duration_seconds=elapsed),
            )
            if args.verbose:
                _log(f"not found ({_format_duration(elapsed)})", level=1)
                _log("")
            else:
                _log(f"{label}  not found ({_format_duration(elapsed)})")
            stats.errors.append(msg)
            continue
        except httpx.HTTPError as exc:
            # Transient network failure (timeout, connection reset, 5xx, ...) that
            # outlasted huggingface_hub's own retries. One bad file/connection
            # shouldn't take down the whole multi-dataset sync - skip this dataset
            # and keep going; re-running the sync later will resume it (ETag-based).
            elapsed = time.perf_counter() - started
            msg = f"skip (network error after retries): {repo_id} ({exc})"
            stats.dataset_results.append(
                DatasetSyncResult(repo_id=repo_id, duration_seconds=elapsed),
            )
            if args.verbose:
                _log(f"network error, skipping ({_format_duration(elapsed)}): {exc}", level=1)
                _log("")
            else:
                _log(f"{label}  network error, skipping ({_format_duration(elapsed)}): {exc}")
            stats.errors.append(msg)
            continue

        elapsed = time.perf_counter() - started
        stats.datasets_synced += 1
        stats.files_downloaded += downloaded
        stats.files_skipped += skipped
        stats.dataset_results.append(
            DatasetSyncResult(
                repo_id=repo_id,
                duration_seconds=elapsed,
                files_downloaded=downloaded,
                files_skipped=skipped,
            ),
        )

        action = "would download" if args.dry_run else "downloaded"
        summary = (
            f"{action} {downloaded}, skipped {skipped}, "
            f"took {_format_duration(elapsed)}"
        )
        if args.verbose:
            _log(summary, level=1)
            _log("")
        else:
            _log(f"{label}  {summary}")

    report = {
        "org": org_slug,
        "local_dir": str(args.local_dir.resolve()),
        "dry_run": args.dry_run,
        "synced_at": now_timestamps(),
        "datasets_synced": stats.datasets_synced,
        "files_downloaded": stats.files_downloaded,
        "files_skipped": stats.files_skipped,
        "errors": stats.errors,
        "datasets": [
            {
                "repo_id": result.repo_id,
                "duration_seconds": round(result.duration_seconds, 3),
                "files_downloaded": result.files_downloaded,
                "files_skipped": result.files_skipped,
            }
            for result in stats.dataset_results
        ],
    }

    args.local_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.local_dir / ".sync_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    _log("")
    _log(
        f"{'Would download' if args.dry_run else 'Downloaded'}: "
        f"{stats.files_downloaded} file(s), "
        f"skipped (unchanged): {stats.files_skipped}, "
        f"datasets: {stats.datasets_synced}, "
        f"errors: {len(stats.errors)}",
    )
    _log(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
