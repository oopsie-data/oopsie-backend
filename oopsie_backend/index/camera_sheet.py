"""Contact sheet of the unified camera names, for checking ``configs/camera_names.yaml``.

Usage:
    uv run oopsie index camera-sheet --data-dir <mirror> --index episode_index.parquet \\
        --out reports/camera_names.html

For every dataset and combination of cameras, the sheet shows the middle frame of
each camera of one episode, labelled with its source camera and unified name.
Combinations with camera name issues come first. Reads the index's
``unified_cameras`` and ``camera_name_issues`` (``oopsie index camera-names``).
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import cv2
import polars as pl

from oopsie_backend.common.repos import DEFAULT_DATA_DIR, episode_path
from oopsie_backend.common.videos import episode_videos
from oopsie_backend.index.store import DEFAULT_INDEX, read_index

COLUMNS = ["dataset_repo", "repo_path", "on_disk", "biarm", "unified_cameras", "camera_name_issues"]
THUMB_HEIGHT = 180

STYLE = """
:root { --bg: #fff; --fg: #1a1a1a; --muted: #666; --line: #ddd; --issue: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #e8e8e8; --muted: #9a9a9a; --line: #333; --issue: #f2b8b5; } }
body { background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; margin: 16px; }
section { border-top: 1px solid var(--line); padding: 12px 0; }
h2 { font-size: 15px; margin: 0 0 4px; }
.meta { color: var(--muted); margin-bottom: 8px; }
.issue { color: var(--issue); }
.row { display: flex; flex-wrap: wrap; gap: 8px; }
figure { margin: 0; }
img { height: 180px; display: block; }
figcaption { font-size: 12px; }
figcaption b { font-weight: 600; }
"""


def middle_frame(path: Path) -> str | None:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (max(1, w * THUMB_HEIGHT // h), THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(jpg.tobytes()).decode() if ok else None


def section(row: dict, data_dir: Path, episodes: int) -> str:
    unified = row["unified_cameras"] or []
    videos = episode_videos(episode_path(data_dir, row["dataset_repo"], row["repo_path"]), row["repo_path"])
    figures = []
    for entry in unified:
        name = entry["name"] or "left out"
        frame = middle_frame(videos[entry["camera"]]) if entry["camera"] in videos else None
        image = f'<img src="data:image/jpeg;base64,{frame}" alt="">' if frame else "<p>no frame</p>"
        caption = f"{html.escape(entry['camera'])} → <b>{html.escape(name)}</b>"
        figures.append(f"<figure>{image}<figcaption>{caption}</figcaption></figure>")
    issues = "".join(f'<div class="issue">{html.escape(i)}</div>' for i in row["camera_name_issues"] or [])
    arms = "bimanual" if row["biarm"] else "single arm"
    return (
        f"<section><h2>{html.escape(row['dataset_repo'])}</h2>"
        f'<div class="meta">{episodes} episode(s), {arms}, e.g. {html.escape(row["repo_path"])}</div>'
        f'{issues}<div class="row">{"".join(figures)}</div></section>'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=Path("reports/camera_names.html"))
    args = parser.parse_args(argv)

    df = read_index(args.index, COLUMNS).filter(pl.col("on_disk") & pl.col("unified_cameras").is_not_null())
    present = [episode_path(args.data_dir, d, r).is_file() for d, r in zip(df["dataset_repo"], df["repo_path"])]
    df = df.filter(pl.Series(present))
    df = df.with_columns(
        pl.col("unified_cameras").list.eval(pl.element().struct.field("camera")).list.sort().list.join(",").alias("_set"),
        pl.col("camera_name_issues").list.len().fill_null(0).alias("_issues"),
    )
    groups = df.group_by("dataset_repo", "_set", "biarm").agg(pl.all().first(), pl.len().alias("_episodes"))
    groups = groups.sort(pl.col("_issues") == 0, "dataset_repo", "_set")
    sections = [section(r, args.data_dir, r["_episodes"]) for r in groups.iter_rows(named=True)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Camera names</title><style>{STYLE}</style></head><body><h1>Camera names</h1>{''.join(sections)}</body></html>"
    )
    print(f"{len(sections)} camera combination(s) in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
