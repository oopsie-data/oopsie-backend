"""Map every episode's policy name to a policy family and class, as defined in ``configs/policies.yaml``.

Usage:
    uv run oopsie index policies --index episode_index.parquet

Reads the index's ``dataset_repo``, ``policy_name`` and ``duplicate_of`` columns and writes,
for every episode, ``policy_family`` (the model, e.g. ``pi0.5`` or ``act``), ``policy_class``
(e.g. ``teleop``, ``frontier`` or ``custom``), ``frontier_group`` (the model line of a
``frontier`` policy, e.g. ``pi``; null for other classes) and ``policy_issues``: policy names
without a mapping, names marked ``unknown`` and frontier families without a group. A duplicate copy is named by the dataset of its original.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

POLICIES_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "policies.yaml"
UNKNOWN = "unknown"
SCHEMA = {
    KEY: pl.String,
    "policy_family": pl.String,
    "policy_class": pl.String,
    "frontier_group": pl.String,
    "policy_issues": pl.List(pl.String),
}
COLUMNS = [c for c in SCHEMA if c != KEY]
INPUTS = ["dataset_repo", "policy_name"]
OPTIONAL = {"duplicate_of": pl.String}


def _rule(value, families: dict) -> tuple[str, str]:
    """``(family, class)`` of a rule given as a family name or as ``{family, class}``."""
    if isinstance(value, dict):
        return value["family"], value.get("class") or families.get(value["family"])
    return value, families.get(value)


def _compile(table: dict, families: dict) -> dict:
    return {
        "aliases": {raw: _rule(v, families) for raw, v in (table.get("aliases") or {}).items()},
        "patterns": [(re.compile(p["match"], re.IGNORECASE), _rule(p, families)) for p in table.get("patterns") or []],
        "default": _rule(table["default"], families) if table.get("default") else None,
    }


def load_config(path: Path = POLICIES_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    classes = set(cfg.get("classes") or [])
    families = cfg.get("families") or {}
    bad = [f"families.{f}: {c}" for f, c in families.items() if c not in classes]
    groups = {}
    for group, members in (cfg.get("frontier_groups") or {}).items():
        for family in members or []:
            if family not in families or family in groups:
                bad.append(f"frontier_groups.{group}: {family}")
            groups[family] = group
    tables = {"": _compile(cfg, families)}
    for dataset, table in (cfg.get("datasets") or {}).items():
        tables[dataset] = _compile(table or {}, families)
    for where, table in tables.items():
        rules = [*table["aliases"].values(), *(r for _, r in table["patterns"]), table["default"]]
        for family, cls in filter(None, rules):
            if family not in families or cls not in classes:
                bad.append(f"{where or 'global'}: {family} / {cls}")
    if bad:
        raise ValueError(f"{path}: not a policy family or class: {sorted(set(bad))}")
    return {"families": families, "groups": groups, "global": tables.pop(""), "datasets": tables}


def _pattern(name: str, table: dict) -> tuple[str, str] | None:
    return next((rule for pattern, rule in table["patterns"] if pattern.search(name)), None)


def unify(dataset: str, name: str | None, cfg: dict) -> dict:
    """Policy columns of one episode; ``dataset`` is the repo name without its namespace."""
    if name is None:
        return _columns(None, None, None, ["no policy recorded"])
    local = cfg["datasets"].get(dataset) or {"aliases": {}, "patterns": [], "default": None}
    glob, families = cfg["global"], cfg["families"]
    rule = (
        local["aliases"].get(name)
        or _pattern(name, local)
        or glob["aliases"].get(name)
        or ((name, families[name]) if name in families else None)
        or _pattern(name, glob)
        or local["default"]
    )
    if rule is None:
        return _columns(None, None, None, [f"no policy family for {name}"])
    family, cls = rule
    issues = [f"policy {name} is marked unknown"] if family == UNKNOWN else []
    group = cfg["groups"].get(family) if cls == "frontier" else None
    if cls == "frontier" and group is None:
        issues.append(f"no frontier group for family {family}")
    return _columns(family, cls, group, issues)


def _columns(family: str | None, cls: str | None, group: str | None, issues: list[str]) -> dict:
    return {"policy_family": family, "policy_class": cls, "frontier_group": group, "policy_issues": issues}


def policies_frame(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    repos = dict(zip(df[KEY].to_list(), df["dataset_repo"].to_list()))
    rows = []
    for r in df.iter_rows(named=True):
        dataset = (repos.get(r["duplicate_of"]) or r["dataset_repo"] or "").rsplit("/", 1)[-1]
        rows.append({KEY: r[KEY], **unify(dataset, r["policy_name"], cfg)})
    return pl.DataFrame(rows, schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=POLICIES_CONFIG)
    args = parser.parse_args(argv)

    frame = policies_frame(read_index(args.index, INPUTS, optional=OPTIONAL), load_config(args.config))
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    issues = frame.select("policy_issues").explode("policy_issues").drop_nulls()
    print(f"{frame.height} episode(s), {issues.height} policy issue(s)", file=sys.stderr)
    for issue, n in issues.group_by("policy_issues").len().sort("len", descending=True).head(20).iter_rows():
        print(f"  {n:>6}  {issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
