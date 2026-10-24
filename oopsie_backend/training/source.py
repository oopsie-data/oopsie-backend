"""Read the arrays, robot profile and videos of a source episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.common.h5 import _profile, _s
from oopsie_backend.common.videos import episode_videos


class Skip(Exception):
    """The episode cannot be unified under the recipe; the message is the reason."""


@dataclass
class SourceEpisode:
    actions: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    profile: dict
    videos: dict[str, Path]
    length: int

    @property
    def rate(self) -> float:
        return float(self.profile.get("control_freq") or 0.0)

    @property
    def arms(self) -> int:
        return 2 if self.profile.get("is_biarm") else 1


def _arrays(group) -> dict[str, np.ndarray]:
    if not isinstance(group, h5py.Group):
        return {}
    out = {}
    for name, ds in group.items():
        if isinstance(ds, h5py.Dataset) and ds.shape is not None and ds.size:
            value = np.asarray(ds, dtype=np.float64)
            out[name] = value.reshape(len(value), -1)
    return out


def read_source(path: Path, repo_path: str, with_videos: bool = True) -> SourceEpisode:
    with h5py.File(path, "r") as f:
        actions = _arrays(f.get("actions"))
        states = _arrays(f.get("observations/robot_states"))
        profile = _profile(_s(f.attrs.get("robot_profile")))
    lengths = {len(v) for v in (*actions.values(), *states.values())}
    if len(lengths) != 1:
        raise Skip(f"arrays have different lengths {sorted(lengths)}")
    episode = SourceEpisode(actions, states, profile, episode_videos(path, repo_path) if with_videos else {}, lengths.pop())
    if episode.rate <= 0:
        raise Skip("no control_freq in the robot profile")
    return episode
