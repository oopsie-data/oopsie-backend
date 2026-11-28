"""Materialize the training data of a recipe (``configs/training/<recipe>.yaml``).

Usage:
    uv run oopsie training materialize --recipe configs/training/droid_joint_velocity.yaml \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --index episode_index.parquet \\
        --out /corral/utexas/IRI26024/training --work-dir $SCRATCH/oopsie/training \\
        --stage-dir $SCRATCH/oopsie/stage

Steps:

1. Select episodes from the index (``training/select.py``).
2. Compute each robot's gripper range from the mirror, used to normalize gripper positions.
3. Unify every selected episode into ``episodes/`` (``training/unify.py``); unified
   episodes that are still current are reused. With ``--stage-dir``, episodes are
   copied to fast storage (``common/staging.py``) one chunk at a time and each
   chunk's copies are removed once it is unified. In a multi-node Slurm job the
   gripper ranges and the unification run on every node (``common/shards.py``,
   shard directory ``<work-dir>/<recipe name>/shards``).
4. Write ``manifest.parquet`` (every selected episode with its status, skip reason
   and labels), ``layout.json`` and ``recipe.yaml``, then each format in ``formats``
   (``training/writers/registry.py``) into ``<format>/``.
5. Set the index's ``training_sets`` column: the recipes each episode is part of.

Everything is built in ``<work-dir>/<recipe name>``. When ``--out`` differs, the
finished dataset is then copied to ``<out>/<recipe name>``, replacing the previous
copy; the unified episodes are copied only for the ``hdf5`` format. Keeping the
work directory on fast scratch storage lets later runs reuse its unified episodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.common import shards
from oopsie_backend.common.pools import process_pool
from oopsie_backend.common.progress import Progress
from oopsie_backend.common.repos import DEFAULT_DATA_DIR, episode_path
from oopsie_backend.common.staging import chunks, stage_episodes, unstage_episodes
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns
from oopsie_backend.training.layout import build_layout
from oopsie_backend.training.recipe import load_recipe, unify_hash
from oopsie_backend.training.select import COLUMNS, OPTIONAL, select_episodes
from oopsie_backend.training.unify import gripper_job, robot_id, unify_episode
from oopsie_backend.training.writers.registry import writer

TRAINING_COLUMN = "training_sets"
TRAINING_SCHEMA = {KEY: pl.String, TRAINING_COLUMN: pl.List(pl.String)}
LABELS = ["episode_id", "lab_id", "robot_name", "language_instruction", "task", "success", "failure_categories"]


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def gripper_extents(jobs: list[tuple[str, Path]], workers: int) -> list[tuple[str, tuple[float, float] | None]]:
    """``(robot, (min, max))`` of the gripper position of every ``(robot, path)``."""
    with process_pool(max_workers=workers) as pool:
        return list(pool.map(gripper_job, jobs, chunksize=16))


def gripper_ranges(extents: list[tuple[str, tuple[float, float] | None]]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for robot, extent in extents:
        if extent is not None:
            lo, hi = ranges.get(robot, extent)
            ranges[robot] = (min(lo, extent[0]), max(hi, extent[1]))
    return ranges


def unify_rows(
    rows: list[dict],
    recipe: dict,
    uhash: str,
    ranges: dict[str, tuple[float, float]],
    data_dir: Path,
    episodes_dir: Path,
    workers: int,
    stage_dir: Path | None,
) -> list[dict]:
    """``unify_episode`` of every row, in order, staging the episodes chunk by chunk with ``stage_dir``."""
    results = []
    progress = Progress(len(rows), "episode(s) unified")
    with process_pool(max_workers=workers) as pool:
        for chunk in chunks(rows) if stage_dir is not None else [rows]:
            episodes = [(r["dataset_repo"], r["repo_path"]) for r in chunk]
            read_dir = stage_episodes(data_dir, stage_dir, episodes, workers) if stage_dir is not None else data_dir
            if not results:
                progress.restart()
            jobs = [
                (
                    r[KEY], r["repo_path"], episode_path(read_dir, r["dataset_repo"], r["repo_path"]), episodes_dir,
                    recipe, uhash, ranges.get(robot_id(r)), r["unified_cameras"],
                )
                for r in chunk
            ]
            try:
                for result in pool.map(unify_episode, jobs, chunksize=1):
                    if result.get("traceback"):
                        _log(result.pop("traceback"))
                    results.append(result)
                    progress.tick(len(results))
            finally:
                if stage_dir is not None:
                    unstage_episodes(data_dir, stage_dir, episodes)
    return results


def run_shard(shard_dir: Path) -> None:
    """One node's share of a sharded phase: gripper extents or unification."""
    job = shards.read_input(shard_dir)
    items = shards.mine(job["items"])
    positions = [i for i, _ in items]
    if job["phase"] == "extents":
        out = gripper_extents([j for _, j in items], job["workers"])
    else:
        out = unify_rows(
            [r for _, r in items], job["recipe"], job["uhash"], job["ranges"], job["data_dir"],
            job["episodes_dir"], job["workers"], job["stage_dir"],
        )
    shards.write_result(shard_dir, dict(zip(positions, out)))


def sharded(shard_dir: Path, phase: str, items: list, **payload) -> list:
    """``phase`` of ``run_shard`` over ``items`` on every node of the job; results in order."""
    shards.clear(shard_dir)
    shards.write_input(shard_dir, {"phase": phase, "items": list(enumerate(items)), **payload})
    rc = shards.run("oopsie_backend.training.materialize", [], shard_dir)
    out: dict[int, object] = {}
    for _, part in shards.results(shard_dir):
        out |= part
    shards.clear(shard_dir)
    if rc or len(out) != len(items):
        raise RuntimeError(f"training shards ({phase}) exited {rc} with {len(out)}/{len(items)} episode(s) done")
    return [out[i] for i in range(len(items))]


def unify_all(
    selected: pl.DataFrame,
    recipe: dict,
    data_dir: Path,
    episodes_dir: Path,
    workers: int,
    stage_dir: Path | None = None,
    shard_dir: Path | None = None,
) -> pl.DataFrame:
    """Unify every selected episode; with ``shard_dir``, on every node of the job.

    The gripper extents pass reads only the gripper datasets, straight from ``data_dir``; staging
    would copy every h5 and its videos for it.
    """
    episodes_dir.mkdir(parents=True, exist_ok=True)
    rows = list(selected.iter_rows(named=True))
    uhash = unify_hash(recipe)
    extent_jobs = [(robot_id(r), episode_path(data_dir, r["dataset_repo"], r["repo_path"])) for r in rows]
    if shard_dir is None:
        ranges = gripper_ranges(gripper_extents(extent_jobs, workers))
        results = unify_rows(rows, recipe, uhash, ranges, data_dir, episodes_dir, workers, stage_dir)
    else:
        ranges = gripper_ranges(sharded(shard_dir, "extents", extent_jobs, workers=workers))
        results = sharded(
            shard_dir, "unify", rows, recipe=recipe, uhash=uhash, ranges=ranges, data_dir=data_dir,
            episodes_dir=episodes_dir, workers=workers, stage_dir=stage_dir,
        )
    manifest = pl.DataFrame(
        [
            {KEY: r[KEY], "dataset_repo": r["dataset_repo"], "repo_path": r["repo_path"], **{c: r[c] for c in LABELS}, **res}
            for r, res in zip(rows, results)
        ],
        schema_overrides={"success": pl.Float64, "failure_categories": pl.List(pl.String)},
        infer_schema_length=None,
    )
    for column in ("num_steps", "fps", "action_widths", "state_widths", "cameras", "actions_from_states", "filled_steps", "reason"):
        if column not in manifest.columns:
            manifest = manifest.with_columns(pl.lit(None).alias(column))
    keep = {f"{k}.h5" for k in manifest.filter(pl.col("status").is_in(["written", "reused"]))[KEY]}
    for path in episodes_dir.glob("*.h5"):
        if path.name not in keep:
            path.unlink()
            shutil.rmtree(episodes_dir / path.stem, ignore_errors=True)
    return manifest


def update_training_sets(index: Path, name: str, included: list[str]) -> None:
    df = read_index(index, [TRAINING_COLUMN], optional={TRAINING_COLUMN: pl.List(pl.String)})
    chosen = set(included)
    rows = []
    for key, sets in zip(df[KEY].to_list(), df[TRAINING_COLUMN].to_list()):
        new = sorted((set(sets or []) - {name}) | ({name} if key in chosen else set()))
        if new != sorted(sets or []):
            rows.append({KEY: key, TRAINING_COLUMN: new or None})
    if rows or TRAINING_COLUMN not in pl.read_parquet_schema(index):
        update_columns(index, pl.DataFrame(rows, schema=TRAINING_SCHEMA), [TRAINING_COLUMN])


def publish(work: Path, dest: Path, formats: list[str]) -> None:
    """Copy the finished dataset from ``work`` to ``dest``, replacing an earlier copy.

    ``rsync -aH`` keeps hard links, so videos the formats share are copied once.
    """
    partial = dest.with_name(f".{dest.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    names = ["manifest.parquet", "layout.json", "recipe.yaml", *("episodes" if f == "hdf5" else f for f in formats)]
    sources = [str(work / n) for n in names if (work / n).exists()]
    subprocess.run(["rsync", "-aH", *sources, f"{partial}/"], check=True)
    old = dest.with_name(f".{dest.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
    partial.rename(dest)
    shutil.rmtree(old, ignore_errors=True)


def materialize(
    recipe_path: Path, data_dir: Path, index: Path, out: Path, work_dir: Path, workers: int, stage_dir: Path | None = None
) -> Path:
    recipe = load_recipe(recipe_path)
    name = recipe["name"]
    work = work_dir / name
    df = read_index(index, COLUMNS, optional=OPTIONAL)
    selected = select_episodes(df, recipe)
    _log(f"{name}: {selected.height} episode(s) selected from {df.height}")
    shard_dir = work / "shards" if shards.node_count() > 1 else None
    manifest = unify_all(selected, recipe, data_dir, work / "episodes", workers, stage_dir, shard_dir)
    included = manifest.filter(pl.col("status").is_in(["written", "reused"]))
    counts = Counter(manifest["status"].to_list())
    _log(f"{name}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    for reason, n in Counter(manifest.filter(pl.col("reason").is_not_null())["reason"].to_list()).most_common(10):
        _log(f"  {n:>6}  {reason}")
    if included.is_empty():
        raise RuntimeError(f"{name}: no episode could be unified")

    layout = build_layout(recipe, included)
    manifest.write_parquet(work / "manifest.parquet")
    (work / "layout.json").write_text(json.dumps(layout, indent=2))
    (work / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    for fmt in recipe["formats"]:
        write = writer(fmt)
        if write is None:
            continue
        target = work / fmt
        shutil.rmtree(target, ignore_errors=True)
        _log(f"{name}: writing {fmt}")
        write(work / "episodes", target, included, layout, recipe)

    dest = out / name
    if dest.resolve() != work.resolve():
        _log(f"{name}: copying to {dest}")
        publish(work, dest, recipe["formats"])
    update_training_sets(index, name, included[KEY].to_list())
    _log(f"{name}: {included.height} episode(s) in {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, help="Where the finished dataset is kept")
    parser.add_argument("--work-dir", type=Path, default=None, help="Where it is built; default: --out")
    parser.add_argument("--stage-dir", type=Path, default=None, help="Copy the selected episodes here chunk by chunk and read them there")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.shard_dir is not None:
        run_shard(args.shard_dir)
        return 0
    if args.recipe is None or args.out is None:
        parser.error("--recipe and --out are required")
    materialize(args.recipe, args.data_dir, args.index, args.out, args.work_dir or args.out, args.workers, args.stage_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
