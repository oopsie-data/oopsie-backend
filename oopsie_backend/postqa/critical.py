"""Mark episodes with critical issues, as defined in ``configs/critical.yaml``.

Usage:
    uv run python -m oopsie_backend.postqa.critical --index episode_index.parquet

Reads the index's ``read_error``, ``has_duplicate_video``, ``qa_error``,
``duration_seconds`` and ``qa_findings`` columns and writes, for every episode,
``critical`` and ``critical_reasons`` (``read_error``, ``duplicate_video``,
``qa_error``, ``short`` or ``qa:<category>/<test>``), replacing earlier values.
``short`` marks episodes shorter than ``min_duration_seconds``. A check in ``qa_checks``
counts on any finding; a check in ``qa_check_min_findings`` counts only when
one finding key (a camera, for video checks) has at least that many findings. Episodes that QA has not
checked can only be critical through the index conditions. Duplicate copies
(``duplicate_of``) are skipped and keep both columns null.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import yaml

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns
from oopsie_backend.qa.checks.registry import all_checks

CRITICAL_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "critical.yaml"
SCHEMA = {KEY: pl.String, "critical": pl.Boolean, "critical_reasons": pl.List(pl.String)}
COLUMNS = ["critical", "critical_reasons"]
INPUTS = ["duplicate_of", "read_error", "has_duplicate_video", "qa_error", "duration_seconds", "qa_findings"]


def load_config(path: Path = CRITICAL_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    checks = set(cfg.get("qa_checks") or [])
    thresholds = cfg.get("qa_check_min_findings") or {}
    unknown = sorted((checks | set(thresholds)) - set(all_checks()))
    if unknown:
        raise ValueError(f"{path}: unknown QA checks {unknown}")
    both = sorted(checks & set(thresholds))
    if both:
        raise ValueError(f"{path}: QA checks in both qa_checks and qa_check_min_findings: {both}")
    bad = sorted(c for c, n in thresholds.items() if not isinstance(n, int) or n < 1)
    if bad:
        raise ValueError(f"{path}: qa_check_min_findings must be positive integers, got {bad}")
    return cfg


def critical_frame(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    checks = set(cfg.get("qa_checks") or [])
    thresholds = cfg.get("qa_check_min_findings") or {}
    rows = []
    for r in df.iter_rows(named=True):
        if r.get("duplicate_of"):
            rows.append({KEY: r[KEY], "critical": None, "critical_reasons": None})
            continue
        reasons = []
        if cfg.get("read_error") and r.get("read_error"):
            reasons.append("read_error")
        if cfg.get("duplicate_video") and r.get("has_duplicate_video"):
            reasons.append("duplicate_video")
        if cfg.get("qa_error") and r.get("qa_error"):
            reasons.append("qa_error")
        min_duration = cfg.get("min_duration_seconds")
        if min_duration is not None and r.get("duration_seconds") is not None and r["duration_seconds"] < min_duration:
            reasons.append("short")
        counts = Counter((f"{f['category']}/{f['test']}", f["key"]) for f in r.get("qa_findings") or [])
        hit = sorted(
            {c for c, _ in counts if c in checks}
            | {c for (c, _), n in counts.items() if c in thresholds and n >= thresholds[c]}
        )
        reasons += [f"qa:{c}" for c in hit]
        rows.append({KEY: r[KEY], "critical": bool(reasons), "critical_reasons": reasons})
    return pl.DataFrame(rows, schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config", type=Path, default=CRITICAL_CONFIG)
    args = parser.parse_args(argv)

    df = read_index(args.index)
    df = df.with_columns(pl.lit(None).alias(c) for c in INPUTS if c not in df.columns)
    frame = critical_frame(df.select(KEY, *INPUTS), load_config(args.config))
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    counts = frame.explode("critical_reasons").group_by("critical_reasons").len().drop_nulls().sort("len", descending=True)
    detail = ", ".join(f"{r} {n}" for r, n in counts.iter_rows())
    print(f"{frame['critical'].sum()} critical episode(s){': ' + detail if detail else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
