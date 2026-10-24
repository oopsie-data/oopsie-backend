"""Find episodes that are copies of each other and mark all but one copy.

Usage:
    uv run python -m oopsie_backend.preqa.duplicate_episodes --index episode_index.parquet

Two on-disk episodes are copies when any of these hold:

- ``identical_data``: the same ``content_hash`` (every action and robot state array equal)
- ``identical_videos``: every camera's video file has the same Hub hash as the other
  episode's, with at least one camera
- ``same_metadata``: same lab, timestamp, duration, instruction, robot and cameras, with
  a recorded timestamp; this catches copies whose arrays were re-exported or trimmed

Copies linked by any rule form one group. The first copy, contributor repos first and
then by repo and path, is kept; every other copy gets ``duplicate_of`` set to its key.
Writes, for every episode, ``duplicate_of``, ``duplicate_reasons`` (the rules that link
it to another copy) and ``duplicate_group`` (the other copies' keys), replacing earlier
values. Downstream steps leave out episodes with ``duplicate_of`` set.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from oopsie_backend.common.repos import CONTRIBUTOR
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

IDENTICAL_DATA = "identical_data"
IDENTICAL_VIDEOS = "identical_videos"
SAME_METADATA = "same_metadata"
SCHEMA = {
    KEY: pl.String,
    "duplicate_of": pl.String,
    "duplicate_reasons": pl.List(pl.String),
    "duplicate_group": pl.List(pl.String),
}
COLUMNS = [c for c in SCHEMA if c != KEY]
INPUTS = [
    "dataset_repo", "repo_path", "repo_kind", "on_disk", "content_hash", "videos",
    "lab_id", "timestamp", "duration_seconds", "language_instruction", "robot_name", "cameras",
]


def _video_signature(videos: list[dict] | None) -> tuple | None:
    hashes = [(v["camera"], v["sha256"] or v["git_oid"]) for v in videos or []]
    if not hashes or any(h is None for _, h in hashes):
        return None
    return tuple(sorted(hashes))


def duplicate_frame(df: pl.DataFrame) -> pl.DataFrame:
    rows = df.filter(pl.col("on_disk")).to_dicts()
    signatures = {
        IDENTICAL_DATA: lambda r: r["content_hash"],
        IDENTICAL_VIDEOS: lambda r: _video_signature(r["videos"]),
        SAME_METADATA: lambda r: (
            r["lab_id"], r["timestamp"], r["duration_seconds"], r["language_instruction"],
            r["robot_name"], tuple(r["cameras"] or ()),
        ) if r["timestamp"] is not None else None,
    }
    parent = {r[KEY]: r[KEY] for r in rows}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    reasons: dict[str, set[str]] = defaultdict(set)
    for reason, sig in signatures.items():
        buckets: dict = defaultdict(list)
        for r in rows:
            s = sig(r)
            if s is not None:
                buckets[s].append(r[KEY])
        for keys in buckets.values():
            if len(keys) < 2:
                continue
            for k in keys:
                reasons[k].add(reason)
                parent[find(k)] = find(keys[0])

    order = {r[KEY]: (r["repo_kind"] != CONTRIBUTOR, r["dataset_repo"], r["repo_path"]) for r in rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for k in parent:
        groups[find(k)].append(k)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=order.__getitem__)
        for k in members:
            out[k] = {
                "duplicate_of": members[0] if k != members[0] else None,
                "duplicate_reasons": sorted(reasons[k]),
                "duplicate_group": [m for m in members if m != k],
            }
    empty = {"duplicate_of": None, "duplicate_reasons": [], "duplicate_group": []}
    return pl.DataFrame([{KEY: k, **out.get(k, empty)} for k in df[KEY].to_list()], schema=SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(argv)

    frame = duplicate_frame(read_index(args.index, columns=INPUTS))
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    copies = frame.filter(pl.col("duplicate_of").is_not_null())
    counts = copies.explode("duplicate_reasons").group_by("duplicate_reasons").len().sort("duplicate_reasons")
    detail = ", ".join(f"{r} {n}" for r, n in counts.iter_rows())
    groups = frame.filter(pl.col("duplicate_group").list.len() > 0).height - copies.height
    print(f"{copies.height} duplicate copies in {groups} group(s){': ' + detail if detail else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
