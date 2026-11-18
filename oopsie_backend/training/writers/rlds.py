"""Export prepared episodes as an RLDS/TFDS dataset at <name>/1.0.0.

Writes images, action/state vectors and masks, language, rewards, episode flags
and provenance. DROID camera/key recipes provide the layout oopsie-rl reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import polars as pl
import tensorflow_datasets as tfds

from oopsie_backend.training.vector_layout import padded

VERSION = tfds.core.Version("1.0.0")


def _frames(path: Path | None, n: int, size: tuple[int, int]) -> Iterator[np.ndarray]:
    blank = np.zeros((*size, 3), dtype=np.uint8)
    capture = cv2.VideoCapture(str(path)) if path else None
    last = blank
    for _ in range(n):
        if capture is not None:
            ok, frame = capture.read()
            if ok:
                last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield last
    if capture is not None:
        capture.release()


def read_example(episodes_dir: Path, row: dict, layout: dict) -> dict[str, Any]:
    size = tuple(layout["image_size"])
    with h5py.File(episodes_dir / f"{row['episode_key']}.h5", "r") as f:
        action, action_mask, action_dict = padded(f["action"], layout["action"])
        state, state_mask, state_dict = padded(f["state"], layout["state"])
        videos = {r: str(f.attrs.get(f"camera/{r}", "")) for r in layout["roles"]}
        fps = float(f.attrs["fps"])
        filled = np.asarray(f["filled"], dtype=bool)
    n = len(action)
    camera_mask = np.array([bool(videos[r]) for r in layout["roles"]])
    images = {r: _frames(episodes_dir / v if v else None, n, size) for r, v in videos.items()}
    success = row["success"]
    succeeded = success is not None and success >= 0.5
    instruction = row["language_instruction"] or ""
    steps = []
    for t in range(n):
        last = t == n - 1
        steps.append(
            {
                "observation": {
                    **{r: next(it) for r, it in images.items()},
                    "state": state[t],
                    **{k: v[t] for k, v in state_dict.items()},
                    "camera_mask": camera_mask,
                },
                "action": action[t],
                "action_dict": {k: v[t] for k, v in action_dict.items()},
                "is_filled": bool(filled[t]),
                "reward": np.float32(1.0 if last and succeeded else 0.0),
                "discount": np.float32(1.0),
                "is_first": t == 0,
                "is_last": last,
                "is_terminal": last and succeeded,
                "language_instruction": instruction,
            }
        )
    outcome = "success" if succeeded else "failure"
    return {
        "steps": steps,
        "episode_metadata": {
            "file_path": f"{outcome}/{row['dataset_repo']}/{row['repo_path']}",
            "episode_key": row["episode_key"],
            "dataset_repo": row["dataset_repo"],
            "repo_path": row["repo_path"],
            "lab_id": row["lab_id"] or "",
            "robot_name": row["robot_name"] or "",
            "fps": np.float32(fps),
            "success": np.float32(-1.0 if success is None else success),
            "failure_categories": json.dumps(row["failure_categories"] or []),
            "action_mask": action_mask,
            "state_mask": state_mask,
        },
    }


class UnifiedRLDSBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = VERSION
    RELEASE_NOTES = {"1.0.0": "Materialized from unified Oopsie episodes."}

    def __init__(self, *, name: str, episodes_dir: Path, included: pl.DataFrame, layout: dict, **kwargs: Any):
        self._episodes_dir, self._rows, self._layout = episodes_dir, list(included.iter_rows(named=True)), layout
        self.name = name
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        h, w = self._layout["image_size"]
        vector = lambda n: tfds.features.Tensor(shape=(n,), dtype=np.float32)  # noqa: E731
        width = lambda part: sum(self._layout[part].values())  # noqa: E731
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                **{r: tfds.features.Image(shape=(h, w, 3), dtype=np.uint8, encoding_format="jpeg") for r in self._layout["roles"]},
                                "state": vector(width("state")),
                                **{k: vector(n) for k, n in self._layout["state"].items()},
                                "camera_mask": tfds.features.Tensor(shape=(len(self._layout["roles"]),), dtype=np.bool_),
                            },
                            "action": vector(width("action")),
                            "action_dict": {k: vector(n) for k, n in self._layout["action"].items()},
                            "is_filled": tfds.features.Scalar(np.bool_),
                            "reward": tfds.features.Scalar(np.float32),
                            "discount": tfds.features.Scalar(np.float32),
                            "is_first": tfds.features.Scalar(np.bool_),
                            "is_last": tfds.features.Scalar(np.bool_),
                            "is_terminal": tfds.features.Scalar(np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": {
                        "file_path": tfds.features.Text(),
                        "episode_key": tfds.features.Text(),
                        "dataset_repo": tfds.features.Text(),
                        "repo_path": tfds.features.Text(),
                        "lab_id": tfds.features.Text(),
                        "robot_name": tfds.features.Text(),
                        "fps": tfds.features.Scalar(np.float32),
                        "success": tfds.features.Scalar(np.float32),
                        "failure_categories": tfds.features.Text(),
                        "action_mask": tfds.features.Tensor(shape=(width("action"),), dtype=np.bool_),
                        "state_mask": tfds.features.Tensor(shape=(width("state"),), dtype=np.bool_),
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://oopsie-data.com/",
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for row in self._rows:
            yield row["episode_key"], read_example(self._episodes_dir, row, self._layout)


def write(episodes_dir: Path, out: Path, included: pl.DataFrame, layout: dict, recipe: dict) -> None:
    builder = UnifiedRLDSBuilder(name=recipe["name"], episodes_dir=episodes_dir, included=included, layout=layout, data_dir=str(out))
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(try_download_gcs=False))
