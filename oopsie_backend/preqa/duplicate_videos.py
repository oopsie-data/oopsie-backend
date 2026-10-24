"""Flag videos whose file is identical to another episode's or another camera's.

Usage:
    uv run python -m oopsie_backend.preqa.duplicate_videos --index episode_index.parquet

Videos are compared by the hash from the Hub listing (``sha256``, else the git
object id). Episodes marked as a duplicate copy by ``preqa.duplicate_episodes``
(``duplicate_of``) and episodes not on disk are left out, since their videos are
expected to repeat.
Videos without a Hub hash cannot be compared and are counted.

Writes ``video_duplicates`` (per duplicated camera: its hash and every other
``episode_key``/camera with the same file) and ``has_duplicate_video`` for every
episode, replacing earlier values.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

VIDEO_DUPLICATE = pl.Struct(
    {"camera": pl.String, "hash": pl.String, "others": pl.List(pl.Struct({KEY: pl.String, "camera": pl.String}))}
)
SCHEMA = {KEY: pl.String, "video_duplicates": pl.List(VIDEO_DUPLICATE), "has_duplicate_video": pl.Boolean}
COLUMNS = ["video_duplicates", "has_duplicate_video"]


def duplicate_videos(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """The duplicate-video columns for every episode, and how many videos had no hash."""
    eligible = df.filter(pl.col("on_disk") & pl.col("duplicate_of").is_null())
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unhashed = 0
    for key, videos in zip(eligible[KEY].to_list(), eligible["videos"].to_list()):
        for v in videos or []:
            h = v["sha256"] or v["git_oid"]
            if h:
                by_hash[h].append((key, v["camera"]))
            else:
                unhashed += 1
    dupes: dict[str, list[dict]] = defaultdict(list)
    for h, uses in by_hash.items():
        if len(uses) < 2:
            continue
        for key, camera in uses:
            others = [{KEY: k, "camera": c} for k, c in uses if (k, c) != (key, camera)]
            dupes[key].append({"camera": camera, "hash": h, "others": others})
    rows = [
        {KEY: k, "video_duplicates": sorted(dupes.get(k, []), key=lambda d: d["camera"]), "has_duplicate_video": k in dupes}
        for k in df[KEY].to_list()
    ]
    return pl.DataFrame(rows, schema=SCHEMA), unhashed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(argv)

    df = read_index(args.index, columns=["on_disk", "videos"], optional={"duplicate_of": pl.String})
    frame, unhashed = duplicate_videos(df)
    update_columns(args.index, frame, COLUMNS, clear_missing=True)
    print(
        f"{frame['has_duplicate_video'].sum()} episode(s) share a video file; "
        f"{unhashed} video(s) without a Hub hash were not compared",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
