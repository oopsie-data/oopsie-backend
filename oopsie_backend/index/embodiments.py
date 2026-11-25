"""Map every episode's robot and gripper to unified names, as defined in ``configs/embodiments.yaml``.

Usage:
    uv run oopsie index embodiments --index episode_index.parquet

Reads the index's ``dataset_repo``, ``robot_name``, ``gripper_name``, ``biarm``,
``uses_mobile_base`` and ``duplicate_of`` columns and writes, for every episode:

- ``unified_arm``: the arm model; two models joined by ``+`` (left, right) for a mixed bimanual setup.
- ``unified_end_effector`` and ``end_effector_type``.
- ``embodiment``: ``<layout>/<arm>/<end effector>``, where layout is ``single_arm`` or
  ``bimanual``, prefixed with ``mobile_`` on a mobile base. Null unless arm and end effector are named.
- ``embodiment_issues``: names without a mapping and names marked ``unknown``.

A duplicate copy is named by the dataset of its original.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

EMBODIMENTS_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "embodiments.yaml"
UNKNOWN = "unknown"
SCHEMA = {
    KEY: pl.String,
    "unified_arm": pl.String,
    "unified_end_effector": pl.String,
    "end_effector_type": pl.String,
    "embodiment": pl.String,
    "embodiment_issues": pl.List(pl.String),
}
COLUMNS = [c for c in SCHEMA if c != KEY]
INPUTS = ["dataset_repo", "robot_name", "gripper_name", "biarm", "uses_mobile_base"]
OPTIONAL = {"duplicate_of": pl.String}


def load_config(path: Path = EMBODIMENTS_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    arms = set(cfg.get("arms") or [])
    end_effectors = cfg.get("end_effectors") or {}
    types = set(cfg.get("end_effector_types") or [])
    bad = [f"end_effectors.{n}: {t}" for n, t in end_effectors.items() if t not in types]
    tables = [("", cfg)] + [(f"datasets.{d}.", t or {}) for d, t in (cfg.get("datasets") or {}).items()]
    for where, table in tables:
        for raw, name in (table.get("arm_aliases") or {}).items():
            names = name if isinstance(name, list) else [name]
            bad += [f"{where}arm_aliases.{raw}: {n}" for n in names if n not in arms and n not in (None, UNKNOWN)]
        for raw, name in (table.get("end_effector_aliases") or {}).items():
            if name not in end_effectors and name not in (None, UNKNOWN):
                bad.append(f"{where}end_effector_aliases.{raw}: {name}")
    if bad:
        raise ValueError(f"{path}: not a unified arm or end effector name: {sorted(bad)}")
    return cfg


def _lookup(raw: str | None, kind: str, known, dataset_table: dict, cfg: dict) -> tuple[str | None, list[str]]:
    """Unified name of one recorded name and the issues it raises."""
    if raw is None:
        return None, [f"no {kind} recorded"]
    key = f"{kind}_aliases"
    for table in (dataset_table.get(key) or {}, cfg.get(key) or {}):
        if raw in table:
            name = table[raw]
            break
    else:
        if raw not in known:
            return None, [f"no unified {kind} for {raw}"]
        name = raw
    if isinstance(name, list):
        name = "+".join(name)
    if name == UNKNOWN:
        return UNKNOWN, [f"{kind} {raw} is marked unknown"]
    return name, []


def unify(
    dataset: str, robot: str | None, gripper: str | None, biarm: bool | None, mobile: bool | None, cfg: dict
) -> dict:
    """Unified embodiment columns of one episode; ``dataset`` is the repo name without its namespace."""
    table = (cfg.get("datasets") or {}).get(dataset) or {}
    end_effectors = cfg.get("end_effectors") or {}
    arm, arm_issues = _lookup(robot, "arm", set(cfg.get("arms") or []), table, cfg)
    effector, effector_issues = _lookup(gripper, "end_effector", end_effectors, table, cfg)
    named = arm not in (None, UNKNOWN) and effector not in (None, UNKNOWN)
    layout = ("mobile_" if mobile else "") + ("bimanual" if biarm else "single_arm")
    return {
        "unified_arm": arm,
        "unified_end_effector": effector,
        "end_effector_type": end_effectors.get(effector),
        "embodiment": f"{layout}/{arm}/{effector}" if named else None,
        "embodiment_issues": arm_issues + effector_issues,
    }


def embodiments_frame(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    repos = dict(zip(df[KEY].to_list(), df["dataset_repo"].to_list()))
    rows = []
    for r in df.iter_rows(named=True):
        dataset = (repos.get(r["duplicate_of"]) or r["dataset_repo"] or "").rsplit("/", 1)[-1]
        unified = unify(dataset, r["robot_name"], r["gripper_name"], r["biarm"], r["uses_mobile_base"], cfg)
        rows.append({KEY: r[KEY], **unified})
    return pl.DataFrame(rows, schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=EMBODIMENTS_CONFIG)
    args = parser.parse_args(argv)

    frame = embodiments_frame(read_index(args.index, INPUTS, optional=OPTIONAL), load_config(args.config))
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    issues = frame.select("embodiment_issues").explode("embodiment_issues").drop_nulls()
    print(f"{frame.height} episode(s), {issues.height} embodiment issue(s)", file=sys.stderr)
    for issue, n in issues.group_by("embodiment_issues").len().sort("len", descending=True).head(20).iter_rows():
        print(f"  {n:>6}  {issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
