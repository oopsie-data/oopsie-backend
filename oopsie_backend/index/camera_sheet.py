"""Contact sheet of the unified camera names, for checking ``configs/camera_names.yaml``.

Usage:
    uv run oopsie index camera-sheet --data-dir <mirror> --index episode_index.parquet \\
        --out reports/camera_names.html

For every dataset and combination of cameras, the sheet shows the middle frame of
each camera of one episode, labelled with its source camera and unified name.
Combinations with camera name issues come first. Reads the index's
``unified_cameras`` and ``camera_name_issues`` (``oopsie index camera-names``).
Pass ``--recipe`` to preview a training recipe's selected camera slots.
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import cv2
import numpy as np
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
.slot { width: min(100%, 320px); }
.slot img { height: auto; width: 100%; }
.padding { background: #000; width: 100%; }
.slot figcaption { overflow-wrap: anywhere; }
.slot figcaption b { display: block; }
"""


def _read_middle_frame(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None


def middle_frame(path: Path, size: tuple[int, int] | None = None) -> str | None:
    frame = _read_middle_frame(path)
    if frame is None:
        from oopsie_backend.training.cameras import opencv_fallback_to_ffmpeg

        with opencv_fallback_to_ffmpeg(path) as decoded:
            frame = _read_middle_frame(decoded) if decoded is not None else None
    if frame is None:
        return None
    h, w = frame.shape[:2]
    height, width = size or (THUMB_HEIGHT, max(1, w * THUMB_HEIGHT // h))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
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


def recipe_section(
    row: dict, videos: dict[str, Path], roles: dict[str, str | None], size: tuple[int, int] | None,
    episodes: int, reason: str | None,
) -> str:
    names = {e["camera"]: e["name"] for e in row["unified_cameras"] or []}
    height, width = size or (THUMB_HEIGHT, 320)
    figures = []
    for role, camera in roles.items():
        frame = middle_frame(videos[camera], size) if camera else None
        if frame:
            image = f'<img loading="lazy" src="data:image/jpeg;base64,{frame}" alt="{html.escape(camera)} middle frame">'
            detail = f'{html.escape(camera)} → {html.escape(names[camera])}'
            if names[camera] == "head" and role.startswith("exterior_"):
                detail += " · head/body fallback"
        else:
            image = f'<div class="padding" style="aspect-ratio:{width}/{height}" role="img" aria-label="Black preview"></div>'
            detail = "Black padding" if camera is None else f"{html.escape(camera)} · preview unavailable"
        figures.append(f'<figure class="slot">{image}<figcaption><b>{html.escape(role)}</b>{detail}</figcaption></figure>')
    unused = [f"{raw} → {names.get(raw) or '(no unified name)'}" for raw in sorted(videos.keys() - set(roles.values()))]
    omitted = f'<div class="meta">Unused views · {html.escape(", ".join(unused))}</div>' if unused else ""
    issue = f'<div class="issue">Skipped · {html.escape(reason)}</div>' if reason else ""
    arms = "bimanual" if row["biarm"] else "single arm"
    return (
        f'<section><h2>{html.escape(row["dataset_repo"])} · {html.escape(row["robot_name"] or "")}</h2>'
        f'<div class="meta">{episodes} episode(s), {arms}, e.g. {html.escape(row["repo_path"])}</div>'
        f'{issue}{omitted}<div class="row">{"".join(figures) if not reason else ""}</div></section>'
    )


def recipe_sheet(index: Path, data_dir: Path, recipe_path: Path) -> tuple[str, str, list[str]]:
    """Preview one representative of each selected source camera and slot combination."""
    from oopsie_backend.training.cameras import assign_episode_roles
    from oopsie_backend.training.read_episode import Skip
    from oopsie_backend.training.recipe import load_recipe
    from oopsie_backend.training.select_episodes import COLUMNS as TRAIN_COLUMNS, OPTIONAL, select_episodes

    recipe = load_recipe(recipe_path)
    selected = select_episodes(read_index(index, TRAIN_COLUMNS, optional=OPTIONAL), recipe)
    cams = recipe["cameras"]
    groups = {}
    mapped = padded = 0
    for row in selected.sort("dataset_repo", "repo_path").iter_rows(named=True):
        path = episode_path(data_dir, row["dataset_repo"], row["repo_path"])
        videos, roles, reason = {}, {}, None
        try:
            videos = episode_videos(path, row["repo_path"])
            roles = assign_episode_roles(row["unified_cameras"], videos, cams)
        except (OSError, Skip) as exc:
            reason = str(exc)
        if reason is None:
            mapped += 1
            padded += None in roles.values()
        names = {item["camera"]: item["name"] for item in row["unified_cameras"] or []}
        signature = (
            row["dataset_repo"], row["robot_name"], row["biarm"], tuple(roles.items()),
            tuple(sorted((source, names.get(source)) for source in videos)), reason,
        )
        group = groups.setdefault(signature, {"row": row, "videos": videos, "roles": roles, "reason": reason, "episodes": 0})
        group["episodes"] += 1
    size = tuple(cams["image_size"]) if cams["image_size"] else None
    sections = [recipe_section(size=size, **g) for g in groups.values()]
    summary = (
        f'<p class="meta">{selected.height} selected episodes · {mapped} with assigned views · '
        f'{padded} with missing slots · {selected.height - mapped} skipped at camera assignment</p>'
    )
    return f'Camera slots for {recipe["name"]}', summary, sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=Path("reports/camera_names.html"))
    parser.add_argument("--recipe", type=Path, help="Show the training recipe's camera slots and missing views")
    args = parser.parse_args(argv)

    if args.recipe:
        title, summary, sections = recipe_sheet(args.index, args.data_dir, args.recipe)
    else:
        title, summary = "Camera names", ""
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
    title = html.escape(title)
    args.out.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style></head><body><h1>{title}</h1>{summary}{''.join(sections)}</body></html>"
    )
    print(f"{len(sections)} camera combination(s) in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
