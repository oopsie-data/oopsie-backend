"""Map every episode's cameras to unified names, as defined in ``configs/camera_names.yaml``.

Usage:
    uv run oopsie index camera-names --index episode_index.parquet

Reads the index's ``dataset_repo``, ``cameras``, ``robot_name``, ``biarm`` and
``duplicate_of`` columns and writes, for every episode, ``unified_cameras`` (``{camera, name}`` per source camera; ``name`` is
null for a camera left out) and ``camera_name_issues``: cameras without a name, cameras
marked ``unknown``, and unified names that two cameras of one episode share. A
duplicate copy is named by the dataset of its original.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

CAMERA_NAMES_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "camera_names.yaml"
UNKNOWN = "unknown"
UNIFIED_TYPE = pl.List(pl.Struct({"camera": pl.String, "name": pl.String}))
SCHEMA = {KEY: pl.String, "unified_cameras": UNIFIED_TYPE, "camera_name_issues": pl.List(pl.String)}
COLUMNS = ["unified_cameras", "camera_name_issues"]
INPUTS = ["dataset_repo", "cameras", "robot_name", "biarm"]
OPTIONAL = {"duplicate_of": pl.String}
WRISTS = ("wrist_left", "wrist_right")


def load_config(path: Path = CAMERA_NAMES_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    names = cfg.get("names") or []
    pattern = re.compile(rf"^({'|'.join(map(re.escape, names))})(_[2-9])?(_stereo)?$")
    tables = {"aliases": cfg.get("aliases") or {}, "biarm_aliases": cfg.get("biarm_aliases") or {}}
    for dataset, table in (cfg.get("datasets") or {}).items():
        table = dict(table or {})
        for robot, robot_table in (table.pop("robots", None) or {}).items():
            tables[f"datasets.{dataset}.robots.{robot}"] = robot_table or {}
        tables[f"datasets.{dataset}"] = table
    bad = sorted(
        f"{where}.{camera}: {name}"
        for where, table in tables.items()
        for camera, name in table.items()
        if name is not None and name != UNKNOWN and not pattern.match(name)
    )
    if bad:
        raise ValueError(f"{path}: not a unified camera name: {bad}")
    return {"pattern": pattern, **cfg}


def unify(
    dataset: str, robot: str | None, cameras: list[str], biarm: bool | None, cfg: dict
) -> tuple[list[dict], list[str]]:
    """``(unified_cameras, issues)`` of one episode; ``dataset`` is the repo name without its namespace."""
    entry = dict((cfg.get("datasets") or {}).get(dataset) or {})
    robots = entry.pop("robots", None) or {}
    tables = [robots.get(robot) or {}, entry]
    if biarm:
        tables.append(cfg.get("biarm_aliases") or {})
    tables.append(cfg.get("aliases") or {})
    out, issues = [], []
    for camera in cameras:
        table = next((t for t in tables if camera in t), None)
        name = table[camera] if table is not None else None
        if table is None:
            issues.append(f"no unified name for camera {camera}")
        elif name == UNKNOWN:
            issues.append(f"camera {camera} is marked unknown")
            name = None
        out.append({"camera": camera, "name": name})
    named = [e for e in out if e["name"]]
    if not biarm and not any(e["name"] == "wrist" for e in named):
        lone = [e for e in named if e["name"] in WRISTS]
        if len(lone) == 1:
            lone[0]["name"] = "wrist"
    for name, n in Counter(e["name"] for e in named).items():
        if n > 1:
            shared = sorted(e["camera"] for e in named if e["name"] == name)
            issues.append(f"cameras {', '.join(shared)} share the name {name}")
    return out, issues


def camera_names_frame(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    repos = dict(zip(df[KEY].to_list(), df["dataset_repo"].to_list()))
    rows = []
    for r in df.iter_rows(named=True):
        repo = repos.get(r["duplicate_of"]) or r["dataset_repo"]
        dataset = (repo or "").rsplit("/", 1)[-1]
        unified, issues = unify(dataset, r["robot_name"], r["cameras"] or [], r["biarm"], cfg)
        rows.append({KEY: r[KEY], "unified_cameras": unified, "camera_name_issues": issues})
    return pl.DataFrame(rows, schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=CAMERA_NAMES_CONFIG)
    args = parser.parse_args(argv)

    frame = camera_names_frame(read_index(args.index, INPUTS, optional=OPTIONAL), load_config(args.config))
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    issues = frame.select("camera_name_issues").explode("camera_name_issues").drop_nulls()
    print(f"{frame.height} episode(s), {issues.height} camera name issue(s)", file=sys.stderr)
    for issue, n in issues.group_by("camera_name_issues").len().sort("len", descending=True).head(20).iter_rows():
        print(f"  {n:>6}  {issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
