"""Build or refresh the core columns of the per-episode parquet index.

Usage:
    uv run python -m oopsie_backend.index.build \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --index episode_index.parquet

Reads h5 metadata of new or changed files only (all files when the index lacks a core column), takes file hashes from the
Hub listing, and marks duplicate copies of the same episode data. Columns
written by other pipeline steps are kept. Episodes whose file disappeared stay
in the index with ``on_disk = false``.

In a multi-node Slurm job the h5 files are read on every node
(``common/shards.py``, shard directory ``<index dir>/shards/index``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from oopsie_backend.common import shards
from oopsie_backend.common.repos import DEFAULT_DATA_DIR
from oopsie_backend.index.core import CORE_SCHEMA, build_core, describe, merge_core, read_jobs
from oopsie_backend.index.store import DEFAULT_INDEX, locked, read_index, write_index


def sharded_reader(shard_dir: Path, argv: list[str]):
    """A ``build_core`` reader that reads the h5 files on every node of the job."""

    def read(jobs: list[tuple[str, Path]], workers: int) -> list[dict]:
        if not jobs:
            return []
        shards.clear(shard_dir)
        shards.write_input(shard_dir, list(enumerate(jobs)))
        rc = shards.run("oopsie_backend.index.build", argv, shard_dir)
        out: dict[int, dict] = {}
        for _, part in shards.results(shard_dir):
            out |= part
        shards.clear(shard_dir)
        if rc or len(out) != len(jobs):
            raise RuntimeError(f"index shards exited {rc} with {len(out)}/{len(jobs)} file(s) read")
        return [out[i] for i in range(len(jobs))]

    return read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--no-hub", action="store_true", help="Skip the Hub listing; keep earlier hashes")
    parser.add_argument("--shard-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.shard_dir is not None:
        jobs = shards.mine(shards.read_input(args.shard_dir))
        shards.write_result(args.shard_dir, dict(zip([i for i, _ in jobs], read_jobs([j for _, j in jobs], args.workers))))
        return 0

    previous = None
    if args.index.is_file():
        present = set(pl.read_parquet_schema(args.index))
        previous = read_index(args.index, columns=[c for c in CORE_SCHEMA if c in present])
    api = None if args.no_hub else HfApi(token=args.token)
    read = read_jobs
    if shards.node_count() > 1:
        read = sharded_reader(args.index.parent / "shards" / "index", ["--workers", str(args.workers)])
    core = build_core(args.data_dir, previous, api, args.workers, read)
    with locked(args.index):
        existing = read_index(args.index) if args.index.is_file() else None
        df = merge_core(core, existing)
        write_index(df, args.index)
    print(f"wrote {args.index}: {describe(df)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
