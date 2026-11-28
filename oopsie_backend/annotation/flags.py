"""Flag problems that the annotators' returns reveal, as defined in ``configs/annotation_flags.yaml``.

Usage:
    uv run oopsie annotation flags --data-dir <mirror> --index episode_index.parquet [--workers N]

Reads each annotated episode's current records (each annotator's latest, made on the
episode's current content) from ``annotation_records`` and writes ``annotation_flags``,
one entry per reason, annotator and source, ``annotation_excluded``, true when a flag's
reason excludes, and ``annotation_unusable``, the annotators whose record a flag's reason
marks unusable for the statistics. The records themselves stay in the index. Note rules match an annotator's ``additional_notes`` or
``episode_description``; the ``idle`` reason reads the robot states of the episode file.
Episodes without annotation records keep both columns null.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import yaml

from oopsie_backend.annotation.columns import current_records
from oopsie_backend.common.pools import process_pool
from oopsie_backend.common.repos import DEFAULT_DATA_DIR, episode_path
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

FLAGS_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "annotation_flags.yaml"
TEXT_FIELDS = ("additional_notes", "episode_description")
IDLE = "idle"
ROBOT_STATE = "robot_state"
FLAG = pl.Struct({"reason": pl.String, "annotator": pl.String, "source": pl.String, "text": pl.String})
SCHEMA = {
    KEY: pl.String,
    "annotation_flags": pl.List(FLAG),
    "annotation_excluded": pl.Boolean,
    "annotation_unusable": pl.List(pl.String),
}
COLUMNS = ["annotation_flags", "annotation_excluded", "annotation_unusable"]
INPUTS = ["dataset_repo", "repo_path", "control_freq", "videos", "annotation_records"]
STATE_KEYS = ("joint_position", "cartesian_position")


class Rule:
    def __init__(self, reason: str, cfg: dict):
        self.reason = reason
        self.exclude = bool(cfg.get("exclude", False))
        self.unusable_record = bool(cfg.get("unusable_record", False))
        self.patterns = [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]]
        self.unless = [re.compile(p, re.IGNORECASE) for p in cfg.get("unless") or []]

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns) and not any(p.search(text) for p in self.unless)


def load_config(path: Path = FLAGS_CONFIG) -> tuple[list[Rule], dict]:
    cfg = yaml.safe_load(path.read_text()) or {}
    rules = [Rule(reason, c) for reason, c in (cfg.get("notes") or {}).items()]
    if IDLE in {r.reason for r in rules}:
        raise ValueError(f"{path}: {IDLE!r} is the robot state reason, not a note rule")
    return rules, cfg[IDLE]


def note_flags(records: list[dict], rules: list[Rule]) -> list[dict]:
    flags = []
    for r in records:
        for source in TEXT_FIELDS:
            text = (r[source] or "").strip()
            if text:
                flags += [
                    {"reason": rule.reason, "annotator": r["annotator"], "source": source, "text": text}
                    for rule in rules
                    if rule.matches(text)
                ]
    return flags


def _positions(path: Path) -> np.ndarray | None:
    with h5py.File(path, "r") as f:
        states = f.get("observations/robot_states")
        name = next((n for n in STATE_KEYS if states is not None and n in states), None)
        if name is None:
            return None
        q = np.asarray(states[name], dtype=np.float64)
    return q.reshape(len(q), -1)


def idle_measures(q: np.ndarray, rate: float, fraction: float) -> tuple[float, float] | None:
    """``(idle share of steps, idle seconds before the first moving step)``."""
    if len(q) < 4 or rate <= 0:
        return None
    speed = np.linalg.norm(np.diff(q, axis=0), axis=1) * rate
    p95 = float(np.quantile(speed, 0.95))
    idle = speed < fraction * p95 if p95 > 0 else np.ones(len(speed), dtype=bool)
    moving = np.flatnonzero(~idle)
    return float(idle.mean()), (int(moving[0]) if len(moving) else len(speed)) / rate


def idle_flag(measures: tuple[float, float] | None, duration: float, cfg: dict) -> str | None:
    """The flag text, or ``None`` when the robot is not idle by ``cfg``."""
    if measures is None:
        return None
    share, start = measures
    parts = []
    if share > cfg["max_idle_share"]:
        parts.append(f"idle in {share:.0%} of steps")
    if start >= cfg["min_idle_start_seconds"] and duration > 0 and start / duration >= cfg["min_idle_start_share"]:
        parts.append(f"idle for the first {start:.1f} s of {duration:.1f} s")
    return "; ".join(parts) or None


def _idle_job(job: tuple[str, Path, float, dict]) -> tuple[str, str | None, str | None]:
    """``(key, flag text, error)`` of one episode file."""
    key, path, rate, cfg = job
    try:
        q = _positions(path)
    except (OSError, KeyError) as exc:
        return key, None, f"{type(exc).__name__}: {exc}"
    if q is None:
        return key, None, "no joint or Cartesian position state"
    return key, idle_flag(idle_measures(q, rate, cfg["idle_speed_fraction"]), len(q) / rate if rate else 0.0, cfg), None


def flags_frame(index: pl.DataFrame, data_dir: Path, rules: list[Rule], idle_cfg: dict, workers: int) -> pl.DataFrame:
    rows = index.filter(pl.col("annotation_records").list.len() > 0)
    excluding = {r.reason for r in rules if r.exclude} | ({IDLE} if idle_cfg.get("exclude") else set())
    unusable = {r.reason for r in rules if r.unusable_record}
    jobs = [
        (r[KEY], episode_path(data_dir, r["dataset_repo"], r["repo_path"]), r["control_freq"] or 0.0, idle_cfg)
        for r in rows.iter_rows(named=True)
    ]
    idle, missing = {}, 0
    with process_pool(max_workers=workers) as pool:
        for key, text, error in pool.map(_idle_job, jobs, chunksize=16):
            missing += error is not None
            if text:
                idle[key] = text
    if missing:
        print(f"  idle: {missing} episode file(s) without readable positions", file=sys.stderr)
    out = []
    for r in rows.iter_rows(named=True):
        records, _ = current_records(r["annotation_records"], r["videos"])
        flags = note_flags(records, rules)
        if r[KEY] in idle:
            flags.append({"reason": IDLE, "annotator": None, "source": ROBOT_STATE, "text": idle[r[KEY]]})
        out.append({
            KEY: r[KEY],
            "annotation_flags": flags,
            "annotation_excluded": any(f["reason"] in excluding for f in flags),
            "annotation_unusable": sorted({f["annotator"] for f in flags if f["reason"] in unusable}),
        })
    return pl.DataFrame(out, schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=FLAGS_CONFIG)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    rules, idle_cfg = load_config(args.config)
    present = set(pl.read_parquet_schema(args.index))
    if "annotation_records" not in present:
        print(f"{args.index} has no annotation columns; run oopsie annotation pipeline index first", file=sys.stderr)
        return 1
    frame = flags_frame(read_index(args.index, INPUTS), args.data_dir, rules, idle_cfg, args.workers)
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    counts = (
        frame.explode("annotation_flags").drop_nulls("annotation_flags")
        .select(KEY, pl.col("annotation_flags").struct.field("reason"))
        .unique().group_by("reason").len().sort("reason")
    )
    for reason, n in counts.iter_rows():
        print(f"  {reason}: {n} episode(s)", file=sys.stderr)
    print(f"flags: {frame['annotation_excluded'].sum()} of {frame.height} annotated episode(s) excluded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
