"""Run the QA checks (``qa/checks/registry.py``) on episodes from the index.

Usage:
    uv run python -m oopsie_backend.qa.run \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --index episode_index.parquet

Each dataset contributes up to ``sampling.default_episode_limit`` episodes, of
which the first ``sampling.video_episode_limit`` also get the video checks
(``configs/action_qa_config.yaml``). The sample is ranked by a hash of the
episode key and ``sampling.random_seed``, so it stays the same as data arrives.
Only on-disk, readable, non-duplicate episodes of contributor repos are sampled,
and of external datasets (``configs/imports.yaml``) with ``scope.include_external_datasets``.

With ``--stage-dir``, the episodes to check and their videos are first copied to
that directory (``common/staging.py``), so the checks read fast local storage.

A check runs on an episode only when the index holds no result of its current
version (``qa/checks/base.py``). Adding, editing or re-configuring a check
therefore reruns just that check; removing one drops its results. Episodes are
read once and checked in parallel. Results go to the index's ``qa_*`` columns,
and ``reports/action_qa.json`` and ``.html`` summarize all current results.

Results are saved to the index every ``run.save_interval`` seconds and when the
run is interrupted (SIGINT or SIGTERM), so a rerun continues where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from oopsie_backend.common.io import now_timestamps, write_stats_json
from oopsie_backend.common.repos import (
    CONTRIBUTOR,
    DEFAULT_DATA_DIR,
    EXTERNAL,
    episode_path,
)
from oopsie_backend.common.staging import stage_episodes
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns
from oopsie_backend.qa.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.qa.action_qa_report import render_action_qa_report
from oopsie_backend.qa.checks.base import VIDEO, Check
from oopsie_backend.qa.checks.registry import enabled_checks
from oopsie_backend.qa.columns import (
    QA_COLUMNS,
    current_entries,
    finding_dict,
    qa_frame,
    qa_row,
    result_entry,
)
from oopsie_backend.qa.engine import now, run_episode

MAX_FLAGGED_DETAIL = 20000


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Progress:
    """Logs ``n/total`` with rate and estimated remaining time at most every ``interval`` seconds."""

    def __init__(self, total: int, interval: float = 30.0):
        self.total, self.interval = total, interval
        self.start = self.last = time.monotonic()

    def tick(self, n: int) -> None:
        t = time.monotonic()
        if n < self.total and t - self.last < self.interval:
            return
        self.last = t
        elapsed = t - self.start
        rate = n / elapsed if elapsed else 0.0
        remaining = (self.total - n) / rate if rate else float("inf")
        _log(
            f"  {n}/{self.total} episode(s) checked ({n / self.total:.0%}), {rate:.2f}/s, "
            f"elapsed {_duration(elapsed)}, remaining ~{_duration(remaining)}"
        )


def _stop_on_sigterm(signum, frame) -> None:
    raise KeyboardInterrupt


def merge_entries(entries: list[dict] | None, new: list[dict]) -> list[dict]:
    """``entries`` with those of the checks in ``new`` replaced."""
    replaced = {e["check"] for e in new}
    return [e for e in entries or [] if e["check"] not in replaced] + new


def save(
    index: Path, results: dict[str, list[dict]], keys, versions: dict[str, str], checks: dict[str, Check]
) -> pl.DataFrame:
    """Write the QA columns of ``keys`` from ``results`` to the index."""
    rows = [qa_row(key, results[key], versions, checks) for key in keys]
    return update_columns(index, qa_frame(rows), QA_COLUMNS)


def select_sample(
    df: pl.DataFrame,
    limit: int | None,
    video_limit: int | None,
    seed: int,
    ranked: bool,
    include_external: bool,
    only_datasets: set[str] | None,
) -> pl.DataFrame:
    """Sampled episodes with their per-dataset ``rank`` and whether they get the video checks."""
    kinds = [CONTRIBUTOR, EXTERNAL] if include_external else [CONTRIBUTOR]
    df = df.filter(
        pl.col("on_disk")
        & pl.col("read_error").is_null()
        & pl.col("duplicate_of").is_null()
        & pl.col("repo_kind").is_in(kinds)
    )
    if only_datasets:
        df = df.filter(pl.col("dataset_repo").is_in(sorted(only_datasets)))
    order = (
        pl.col(KEY).map_elements(lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest(), return_dtype=pl.String)
        if ranked
        else pl.col("repo_path")
    )
    df = (
        df.with_columns(order.alias("_order"))
        .sort("dataset_repo", "_order")
        .with_columns(pl.int_range(pl.len()).over("dataset_repo").alias("rank"))
        .drop("_order")
    )
    if limit is not None:
        df = df.filter(pl.col("rank") < limit)
    return df.with_columns((pl.lit(video_limit is None) | (pl.col("rank") < (video_limit or 0))).alias("video"))


def pending_checks(
    sample: pl.DataFrame, checks: dict[str, Check], versions: dict[str, str], force: bool, retry_errors: bool
) -> dict[str, list[str]]:
    """Episode key -> checks that have no current result."""
    out: dict[str, list[str]] = {}
    results = sample["qa_results"].to_list() if "qa_results" in sample.columns else [None] * sample.height
    for key, video, entries in zip(sample[KEY], sample["video"], results):
        have = {
            e["check"]: e for e in current_entries(entries, versions) if not (retry_errors and e["error"])
        }
        names = [
            n for n, c in checks.items()
            if (c.kind != VIDEO or video) and (force or n not in have)
        ]
        if names:
            out[key] = names
    return out


def build_report(df: pl.DataFrame, checks: dict[str, Check], versions: dict[str, str], scope: str, config: dict) -> dict:
    """The ``action_qa.json`` summary of every current result in the index."""
    tests: dict[str, list[str]] = defaultdict(list)
    for c in checks.values():
        tests[c.category].append(c.test)
    totals: Counter = Counter()
    per_dataset: dict[str, dict] = {}
    flagged: list[dict] = []
    rows = df.filter(pl.col("qa_results").list.len() > 0) if "qa_results" in df.columns else df.clear()
    for r in rows.sort("dataset_repo", "repo_path").iter_rows(named=True):
        current = current_entries(r["qa_results"], versions)
        if not current:
            continue
        ds = per_dataset.setdefault(
            r["dataset_repo"],
            {
                "episodes_checked": 0,
                "errors": 0,
                "episodes_flagged": 0,
                "tests": {
                    cat: {
                        t: {"checked_episodes": 0, "flagged_episodes": 0, "flagged_keys": Counter(), "skips": Counter()}
                        for t in ts
                    }
                    for cat, ts in tests.items()
                },
            },
        )
        ds["episodes_checked"] += 1
        if any(e["error"] for e in current):
            ds["errors"] += 1
        findings = []
        for e in current:
            cat, test = e["check"].split("/")
            stats = ds["tests"][cat][test]
            stats["checked_episodes"] += e["ran"]
            stats["flagged_episodes"] += bool(e["findings"])
            stats["flagged_keys"].update(f["key"] for f in e["findings"])
            stats["skips"].update(s["reason"] for s in e["skips"])
            findings += [finding_dict(f) for f in e["findings"]]
        if findings:
            ds["episodes_flagged"] += 1
            flagged.append(
                {
                    "dataset_repo": r["dataset_repo"],
                    "repo_path": r["repo_path"],
                    "episode_id": Path(r["repo_path"]).stem,
                    "findings": findings,
                }
            )
    for ds in per_dataset.values():
        totals["episodes_checked"] += ds["episodes_checked"]
        totals["episodes_skipped"] += ds["errors"]
        totals["episodes_flagged"] += ds["episodes_flagged"]
        for cat in ds["tests"].values():
            for stats in cat.values():
                stats["flagged_keys"] = dict(stats["flagged_keys"])
                stats["skips"] = dict(stats["skips"])
    return {
        "schema_version": 3,
        "generated_at": now_timestamps(),
        "scope": scope,
        "config_snapshot": {k: v for k, v in config.items() if k != "run"} | {"check_versions": versions},
        "totals": dict(totals),
        "per_dataset": per_dataset,
        "flagged_episodes": flagged[:MAX_FLAGGED_DETAIL],
    }


def main(argv: list[str] | None = None) -> int:
    config = ACTION_QA_CONFIG
    run_cfg, sampling = config["run"], config["sampling"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="Default: run.data_dir, else the stats default")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=None, help="Episodes per dataset; 0 means all")
    parser.add_argument("--video-limit", type=int, default=None, help="Of those, episodes with video checks; 0 means all")
    parser.add_argument("--max-episodes", type=int, default=None, help="Check at most this many episodes this run")
    parser.add_argument("--only-dataset", action="append", default=None, metavar="REPO")
    parser.add_argument("--check", action="append", default=None, metavar="CATEGORY/TEST", help="Run only these checks")
    parser.add_argument("--force", action="store_true", help="Rerun the selected checks even if their results are current")
    parser.add_argument("--retry-errors", action="store_true", help="Rerun checks whose current result is an error")
    parser.add_argument("--output", type=Path, default=None, help="Default: run.output")
    parser.add_argument("--workers", type=int, default=None, help="Default: run.workers")
    parser.add_argument("--save-interval", type=float, default=None, help="Seconds between saves to the index; default: run.save_interval")
    parser.add_argument("--stage-dir", type=Path, default=None, help="Copy the checked episodes here first and read them there")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir or run_cfg["data_dir"] or DEFAULT_DATA_DIR)
    output = Path(args.output or run_cfg["output"])
    workers = args.workers or run_cfg["workers"]
    save_interval = args.save_interval if args.save_interval is not None else run_cfg["save_interval"]
    limit = args.limit if args.limit is not None else sampling["default_episode_limit"]
    video_limit = args.video_limit if args.video_limit is not None else sampling["video_episode_limit"]
    only = args.only_dataset if args.only_dataset is not None else run_cfg["only_dataset"]
    max_episodes = args.max_episodes if args.max_episodes is not None else run_cfg["max_total_episodes"]

    checks = enabled_checks(config, args.check)
    versions = {n: c.version(config) for n, c in enabled_checks(config).items()}
    df = read_index(args.index, optional={"duplicate_of": pl.String})
    sample = select_sample(
        df, limit or None, video_limit or None, sampling["random_seed"] or 0, sampling["random"],
        config["scope"]["include_external_datasets"], set(only) if only else None,
    )
    pending = pending_checks(sample, checks, versions, args.force, args.retry_errors)
    keys = [k for k in sample.sort("rank", "dataset_repo")[KEY] if k in pending]
    if max_episodes:
        keys = keys[:max_episodes]
    _log(
        f"{sample.height} sampled episode(s) across {sample['dataset_repo'].n_unique()} dataset(s); "
        f"{len(keys)} with pending checks ({sum(len(pending[k]) for k in keys)} check runs)"
    )

    rows = {r[KEY]: r for r in sample.select(KEY, "dataset_repo", "repo_path").iter_rows(named=True)}
    read_dir = data_dir
    if args.stage_dir and keys:
        read_dir = stage_episodes(data_dir, args.stage_dir, [(rows[k]["dataset_repo"], rows[k]["repo_path"]) for k in keys], workers)
    jobs = [
        (k, episode_path(read_dir, rows[k]["dataset_repo"], rows[k]["repo_path"]), rows[k]["repo_path"], pending[k], config)
        for k in keys
    ]
    all_checks = enabled_checks(config)
    entries = dict(zip(df[KEY].to_list(), df["qa_results"].to_list())) if "qa_results" in df.columns else {}
    unsaved: set[str] = set()
    run_at = now()
    progress = Progress(len(jobs))
    last_save = time.monotonic()
    finished = False
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous_handler = signal.signal(signal.SIGTERM, _stop_on_sigterm) if on_main_thread else None
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        for i, (key, results, _) in enumerate(pool.map(run_episode, jobs, chunksize=2), 1):
            new = [result_entry(n, versions[n], run_at, r) for n, r in results.items()]
            entries[key] = merge_entries(entries.get(key), new)
            unsaved.add(key)
            progress.tick(i)
            if time.monotonic() - last_save >= save_interval and i < len(jobs):
                save(args.index, entries, unsaved, versions, all_checks)
                unsaved.clear()
                last_save = time.monotonic()
        finished = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous_handler or signal.SIG_DFL)
        if not finished and unsaved:
            save(args.index, entries, unsaved, versions, all_checks)
            _log(f"interrupted; saved the results of {len(unsaved)} more episode(s) to {args.index}")

    # Every episode with results, so dropped checks and version changes reach the summary columns.
    touched = [k for k, v in entries.items() if v]
    if touched:
        df = save(args.index, entries, touched, versions, all_checks)

    report = build_report(df, all_checks, versions, f"index:{args.index}", config)
    write_stats_json(output, report)
    render_action_qa_report(report, output.with_suffix(".html"), config["report"]["highlight_ignore"])
    t = report["totals"]
    _log(
        f"{t.get('episodes_checked', 0)} episode(s) with results: {t.get('episodes_flagged', 0)} flagged, "
        f"{t.get('episodes_skipped', 0)} with errors. Report: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
