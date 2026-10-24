"""Write a materialized dataset in the LeRobot v2.1 format.

Layout::

    lerobot/meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
    lerobot/data/chunk-000/episode_000000.parquet
    lerobot/videos/chunk-000/observation.images.<role>/episode_000000.mp4

Frames hold ``observation.state``, ``action``, their masks, ``timestamp``,
``frame_index``, ``episode_index``, ``index`` and ``task_index``. Episode
labels (``success``, ``failure_categories``) and the source episode are extra
fields of ``episodes.jsonl``. The unified videos are hard-linked (or copied);
a blank role gets a black video. LeRobot needs one frame rate, so the recipe
must set ``rate_hz`` unless every episode has the same rate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import polars as pl

from oopsie_backend.training.cameras import FFMPEG
from oopsie_backend.training.layout import dim_names, padded

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
STATS_FRAMES = 16


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _black_video(dst: Path, size: tuple[int, int], fps: float, frames: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = size
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps:g}",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )


def _stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _image_stats(path: Path, frames: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    picked = set(np.linspace(0, frames - 1, min(STATS_FRAMES, frames)).astype(int).tolist())
    samples = []
    for i in range(frames):
        ok, frame = capture.read()
        if not ok:
            break
        if i in picked:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0)
    capture.release()
    pixels = np.concatenate(samples) if samples else np.zeros((1, 3))
    as_image = lambda v: [[[float(x)]] for x in v]  # noqa: E731 - (C, 1, 1), as LeRobot stores image stats
    return {
        "min": as_image(pixels.min(axis=0)),
        "max": as_image(pixels.max(axis=0)),
        "mean": as_image(pixels.mean(axis=0)),
        "std": as_image(pixels.std(axis=0)),
        "count": [len(samples)],
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write(episodes_dir: Path, out: Path, included: pl.DataFrame, layout: dict, recipe: dict) -> None:
    fps = layout["fps"]
    if fps is None:
        raise ValueError("LeRobot needs a single frame rate; set rate_hz in the recipe")
    size = tuple(layout["image_size"])
    roles = layout["roles"]
    (out / "meta").mkdir(parents=True)
    tasks: dict[str, int] = {}
    episodes, episodes_stats = [], []
    index = 0
    for ep_index, row in enumerate(included.iter_rows(named=True)):
        chunk = ep_index // CHUNK_SIZE
        task = row["language_instruction"] or ""
        task_index = tasks.setdefault(task, len(tasks))
        with h5py.File(episodes_dir / f"{row['episode_key']}.h5", "r") as f:
            action, action_mask, _ = padded(f["action"], layout["action"])
            state, state_mask, _ = padded(f["state"], layout["state"])
            videos = {r: str(f.attrs.get(f"camera/{r}", "")) for r in roles}
        n = len(action)
        frames = pl.DataFrame(
            {
                "observation.state": state.tolist(),
                "observation.state_mask": [state_mask.tolist()] * n,
                "action": action.tolist(),
                "action_mask": [action_mask.tolist()] * n,
                "timestamp": (np.arange(n) / fps).astype(np.float32),
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, ep_index, dtype=np.int64),
                "index": np.arange(index, index + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            },
            schema_overrides={
                "observation.state": pl.List(pl.Float32),
                "action": pl.List(pl.Float32),
                "observation.state_mask": pl.List(pl.Boolean),
                "action_mask": pl.List(pl.Boolean),
            },
        )
        data_path = out / f"data/chunk-{chunk:03d}/episode_{ep_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frames.write_parquet(data_path)
        stats = {
            "observation.state": _stats(state),
            "action": _stats(action),
            "timestamp": _stats(frames["timestamp"].to_numpy()[:, None]),
            "frame_index": _stats(frames["frame_index"].to_numpy()[:, None]),
            "episode_index": _stats(frames["episode_index"].to_numpy()[:, None]),
            "index": _stats(frames["index"].to_numpy()[:, None]),
            "task_index": _stats(frames["task_index"].to_numpy()[:, None]),
        }
        for role in roles:
            dst = out / f"videos/chunk-{chunk:03d}/observation.images.{role}/episode_{ep_index:06d}.mp4"
            if videos[role]:
                _link(episodes_dir / videos[role], dst)
            else:
                _black_video(dst, size, fps, n)
            stats[f"observation.images.{role}"] = _image_stats(dst, n)
        episodes.append(
            {
                "episode_index": ep_index,
                "tasks": [task],
                "length": n,
                "episode_key": row["episode_key"],
                "dataset_repo": row["dataset_repo"],
                "repo_path": row["repo_path"],
                "lab_id": row["lab_id"],
                "robot_name": row["robot_name"],
                "success": row["success"],
                "failure_categories": row["failure_categories"],
            }
        )
        episodes_stats.append({"episode_index": ep_index, "stats": stats})
        index += n

    h, w = size
    video_info = {
        "video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False,
    }
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}  # noqa: E731
    features = {
        **{
            f"observation.images.{r}": {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"], "info": video_info}
            for r in roles
        },
        "observation.state": {"dtype": "float32", "shape": [len(dim_names(layout["state"]))], "names": dim_names(layout["state"])},
        "observation.state_mask": {"dtype": "bool", "shape": [len(dim_names(layout["state"]))], "names": dim_names(layout["state"])},
        "action": {"dtype": "float32", "shape": [len(dim_names(layout["action"]))], "names": dim_names(layout["action"])},
        "action_mask": {"dtype": "bool", "shape": [len(dim_names(layout["action"]))], "names": dim_names(layout["action"])},
        "timestamp": scalar("float32"),
        "frame_index": scalar("int64"),
        "episode_index": scalar("int64"),
        "index": scalar("int64"),
        "task_index": scalar("int64"),
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "mixed",
        "total_episodes": len(episodes),
        "total_frames": index,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(roles),
        "total_chunks": (len(episodes) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))
    _jsonl(out / "meta/tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    _jsonl(out / "meta/episodes.jsonl", episodes)
    _jsonl(out / "meta/episodes_stats.jsonl", episodes_stats)
