"""Per-episode work split across the nodes of a multi-node Slurm job.

A step that reads every episode (the index build, QA, training) runs as a
coordinator when the job holds more than one node: it writes the step's input
to a shard directory on shared storage, starts one task per node with
``srun``, and merges the tasks' results. Each task runs the step's module with
``--shard-dir``, takes every ``n``-th item of the input (``mine``) and writes
its result to ``<shard dir>/<task>.pkl``. Only the coordinator writes the
episode index.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

INPUT = "input.pkl"


def node_count() -> int:
    """Nodes of the current Slurm job; 1 outside of one."""
    return int(os.environ.get("SLURM_JOB_NUM_NODES") or 1)


def task() -> tuple[int, int]:
    """``(index, count)`` of this shard task."""
    return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])


def mine(items: list) -> list:
    """This task's share of ``items``."""
    i, n = task()
    return items[i::n]


def _dump(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def write_input(shard_dir: Path, obj) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    _dump(obj, shard_dir / INPUT)


def read_input(shard_dir: Path):
    return _load(shard_dir / INPUT)


def write_result(shard_dir: Path, obj) -> None:
    """Write (or replace) this task's result."""
    i, _ = task()
    _dump(obj, shard_dir / f"{i}.pkl")


def results(shard_dir: Path) -> list[tuple[Path, object]]:
    """Every task result in ``shard_dir`` with its file."""
    if not shard_dir.is_dir():
        return []
    return [(p, _load(p)) for p in sorted(shard_dir.glob("[0-9]*.pkl"))]


def clear(shard_dir: Path) -> None:
    """Remove the input and every task result."""
    for p in [shard_dir / INPUT, *shard_dir.glob("[0-9]*.pkl")]:
        p.unlink(missing_ok=True)


def run(module: str, args: list[str], shard_dir: Path) -> int:
    """Run ``python -m module args --shard-dir shard_dir`` once on every node of the job."""
    n = node_count()
    shard_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "srun", "--overlap", "--label", f"--nodes={n}", f"--ntasks={n}", "--ntasks-per-node=1",
        sys.executable, "-m", module, *args, "--shard-dir", str(shard_dir),
    ]
    print(f"running {module} on {n} node(s)", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode
