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

With ``--stage-dir``, the episodes to check and their videos are copied to that
directory (``common/staging.py``), so the checks read fast local storage. They
are staged in chunks of ``STAGE_CHUNK``: the next chunk is copied while the
current one is checked, and a chunk's copies are removed once it is done or the
run stops, so at most two chunks occupy the staging directory.

A check runs on an episode only when the index holds no result of its current
version (``qa/checks/base.py``). Adding, editing or re-configuring a check
therefore reruns just that check; removing one drops its results. Episodes are
read once and checked in parallel. Results go to the index's ``qa_*`` columns,
and ``reports/action_qa.html`` summarizes all current results.

Results are saved to the index every ``run.save_interval`` seconds, after each
staged chunk and when the run is interrupted (SIGINT or SIGTERM), so a rerun
continues where it stopped.

In a multi-node Slurm job every node checks its share of the pending episodes
(``common/shards.py``) and saves its results to ``<index dir>/shards/qa``
instead of the index. The coordinator merges those files into the index before
and after the run, so results of an interrupted run are kept, and then writes
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import polars as pl

from oopsie_backend.common import interrupt, shards
from oopsie_backend.common.progress import Progress
from oopsie_backend.common.repos import (
    CONTRIBUTOR,
    DEFAULT_DATA_DIR,
    EXTERNAL,
    episode_path,
)
from oopsie_backend.common.staging import chunks as split_chunks
from oopsie_backend.common.staging import stage_episodes, unstage_episodes
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns
from oopsie_backend.qa.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.qa.action_qa_report import render_action_qa_report
from oopsie_backend.qa.checks.base import VIDEO, Check
from oopsie_backend.qa.checks.registry import enabled_checks
from oopsie_backend.qa.columns import (
    QA_COLUMNS,
    current_entries,
    qa_frame,
    qa_row,
    result_entry,
)
from oopsie_backend.qa.engine import now, run_episode
from oopsie_backend.qa.report import build_report

def _log(message: str) -> None:
    print(message, file=sys.stderr)


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


def merge_shards(index: Path, shard_dir: Path, versions: dict[str, str], checks: dict[str, Check]) -> int:
    """Save the results in ``shard_dir`` to the index and remove them; returns the number of episodes."""
    found = shards.results(shard_dir)
    entries: dict[str, list[dict]] = {}
    for _, part in found:
        entries |= part
    if entries:
        save(index, entries, list(entries), versions, checks)
    for path, _ in found:
        path.unlink()
    return len(entries)


def finalize(index: Path, output: Path, config: dict, versions: dict[str, str], checks: dict[str, Check]) -> None:
    """Refresh the QA summary columns of every episode with results and write the report."""
    df = read_index(index)
    if "qa_results" in df.columns:
        # Every episode with results, so dropped checks and version changes reach the summary columns.
        entries = dict(zip(df[KEY].to_list(), df["qa_results"].to_list()))
        touched = [k for k, v in entries.items() if v]
        if touched:
            df = save(index, entries, touched, versions, checks)
    report = build_report(df, checks, versions, f"index:{index}", config)
    render_action_qa_report(report, output, config["report"]["highlight_ignore"])
    t = report["totals"]
    _log(
        f"{t.get('episodes_checked', 0)} episode(s) with results: {t.get('episodes_flagged', 0)} flagged, "
        f"{t.get('episodes_skipped', 0)} with errors. Report: {output}"
    )


def coordinate(argv: list[str], index: Path, output: Path, config: dict) -> int:
    """Run the checks on every node of the job and merge their results."""
    all_checks = enabled_checks(config)
    versions = {n: c.version(config) for n, c in all_checks.items()}
    shard_dir = index.parent / "shards" / "qa"
    if n := merge_shards(index, shard_dir, versions, all_checks):
        _log(f"merged the results of {n} episode(s) from an earlier run")
    rc = shards.run("oopsie_backend.qa.run", argv, shard_dir)
    n = merge_shards(index, shard_dir, versions, all_checks)
    _log(f"merged the results of {n} episode(s) from {shards.node_count()} node(s)")
    finalize(index, output, config, versions, all_checks)
    return rc


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
    parser.add_argument("--shard-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    interrupt.install()

    data_dir = Path(args.data_dir or run_cfg["data_dir"] or DEFAULT_DATA_DIR)
    output = Path(args.output or run_cfg["output"])
    workers = args.workers or run_cfg["workers"]
    save_interval = args.save_interval if args.save_interval is not None else run_cfg["save_interval"]
    limit = args.limit if args.limit is not None else sampling["default_episode_limit"]
    video_limit = args.video_limit if args.video_limit is not None else sampling["video_episode_limit"]
    only = args.only_dataset if args.only_dataset is not None else run_cfg["only_dataset"]
    max_episodes = args.max_episodes if args.max_episodes is not None else run_cfg["max_total_episodes"]
    if args.shard_dir is None and shards.node_count() > 1:
        return coordinate(list(argv if argv is not None else sys.argv[1:]), args.index, output, config)

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
    if args.shard_dir is not None:
        keys = shards.mine(keys)
    _log(
        f"{sample.height} sampled episode(s) across {sample['dataset_repo'].n_unique()} dataset(s); "
        f"{len(keys)} with pending checks ({sum(len(pending[k]) for k in keys)} check runs)"
    )

    rows = {r[KEY]: r for r in sample.select(KEY, "dataset_repo", "repo_path").iter_rows(named=True)}

    def episodes(chunk: list[str]) -> list[tuple[str, str]]:
        return [(rows[k]["dataset_repo"], rows[k]["repo_path"]) for k in chunk]

    def jobs(chunk: list[str], read_dir: Path) -> list[tuple]:
        return [
            (k, episode_path(read_dir, rows[k]["dataset_repo"], rows[k]["repo_path"]), rows[k]["repo_path"], pending[k], config)
            for k in chunk
        ]

    stage_dir = args.stage_dir
    chunks = split_chunks(keys) if stage_dir else [keys]
    all_checks = enabled_checks(config)
    entries = dict(zip(df[KEY].to_list(), df["qa_results"].to_list())) if "qa_results" in df.columns else {}
    unsaved: set[str] = set()
    ran: set[str] = set()

    def persist() -> None:
        """Save the unsaved results: to the index, or as this shard's result."""
        ran.update(unsaved)
        if args.shard_dir is not None:
            shards.write_result(args.shard_dir, {k: entries[k] for k in ran})
        else:
            save(args.index, entries, unsaved, versions, all_checks)
        unsaved.clear()

    run_at = now()
    progress = Progress(len(keys), "episode(s) checked")
    last_save = time.monotonic()
    finished = False
    pool = ProcessPoolExecutor(max_workers=workers)
    stager = ThreadPoolExecutor(max_workers=1)

    def stage(chunk: list[str]):
        return stager.submit(stage_episodes, data_dir, stage_dir, episodes(chunk), workers, interrupt.stopping)

    # Chunks whose episodes are checked from the staging directory, each with its unfinished jobs. The
    # next chunk is submitted as soon as it is staged, so workers never wait for a chunk to drain; one
    # chunk is staged ahead, so at most two chunks are on disk.
    open_jobs: dict[int, int] = {}
    chunk_of: dict = {}
    staged = None
    next_chunk = 0
    with interrupt.stoppable():
        try:
            if not stage_dir:
                chunk_of = {pool.submit(run_episode, job): 0 for job in jobs(keys, data_dir)}
                open_jobs[0] = len(chunk_of)
                next_chunk = len(chunks)
            done_count = 0
            while chunk_of or staged is not None or next_chunk < len(chunks):
                if staged is None and next_chunk < len(chunks) and len(open_jobs) < 2:
                    staged = stage(chunks[next_chunk])
                waiting = set(chunk_of) | ({staged} if staged is not None else set())
                done, _ = wait(waiting, timeout=1.0, return_when=FIRST_COMPLETED)
                if staged is not None and staged in done:
                    read_dir = staged.result()
                    staged = None
                    chunk = chunks[next_chunk]
                    _log(f"chunk {next_chunk + 1}/{len(chunks)}: {len(chunk)} episode(s)")
                    if not chunk_of and not done_count:
                        progress.restart()
                    chunk_of |= {pool.submit(run_episode, job): next_chunk for job in jobs(chunk, read_dir)}
                    open_jobs[next_chunk] = len(chunk)
                    next_chunk += 1
                for future in done & set(chunk_of):
                    key, results, _ = future.result()
                    new = [result_entry(n, versions[n], run_at, r) for n, r in results.items()]
                    entries[key] = merge_entries(entries.get(key), new)
                    unsaved.add(key)
                    done_count += 1
                    progress.tick(done_count)
                    i = chunk_of.pop(future)
                    open_jobs[i] -= 1
                    if not open_jobs[i]:
                        del open_jobs[i]
                        if stage_dir:
                            persist()
                            last_save = time.monotonic()
                            unstage_episodes(data_dir, stage_dir, episodes(chunks[i]))
                if interrupt.stopping():
                    raise KeyboardInterrupt
                if unsaved and time.monotonic() - last_save >= save_interval:
                    persist()
                    last_save = time.monotonic()
            finished = True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            if not finished and unsaved:
                n = len(unsaved)
                persist()
                _log(f"interrupted; saved the results of {n} more episode(s)")
            stager.shutdown(wait=True, cancel_futures=True)
            if not finished and stage_dir:
                for i in [*open_jobs, *([next_chunk] if next_chunk < len(chunks) else [])]:
                    unstage_episodes(data_dir, stage_dir, episodes(chunks[i]))

    if unsaved:
        persist()
    if args.shard_dir is None:
        finalize(args.index, output, config, versions, all_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
