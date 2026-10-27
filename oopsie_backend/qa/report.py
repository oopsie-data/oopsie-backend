"""QA report data built from the index's ``qa_*`` columns."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from oopsie_backend.common.io import now_timestamps
from oopsie_backend.index.store import read_index
from oopsie_backend.qa.checks.base import Check
from oopsie_backend.qa.checks.registry import enabled_checks
from oopsie_backend.qa.columns import current_entries, finding_dict

MAX_FLAGGED_DETAIL = 20000


def build_report(df: pl.DataFrame, checks: dict[str, Check], versions: dict[str, str], scope: str, config: dict) -> dict:
    """Summary of every current QA result in ``df``, as rendered by ``action_qa_report``."""
    tests: dict[str, list[str]] = defaultdict(list)
    for c in checks.values():
        tests[c.category].append(c.test)
    totals: Counter = Counter()
    per_dataset: dict[str, dict] = {}
    flagged: list[dict] = []
    cols = ["dataset_repo", "repo_path", "qa_results"]
    rows = df.select(cols).filter(pl.col("qa_results").list.len() > 0) if "qa_results" in df.columns else df.clear()
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
            if e["findings"]:
                stats["flagged_episodes"] += 1
                stats["flagged_keys"].update(f["key"] for f in e["findings"])
                findings += [finding_dict(f) for f in e["findings"]]
            if e["skips"]:
                stats["skips"].update(s["reason"] for s in e["skips"])
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


def index_report(index: Path, config: dict) -> dict:
    """The report of every current result in ``index``."""
    checks = enabled_checks(config)
    versions = {n: c.version(config) for n, c in checks.items()}
    return build_report(read_index(index), checks, versions, f"index:{index}", config)
