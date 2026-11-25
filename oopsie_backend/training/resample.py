"""Resample action/state arrays to a target rate or preserve the native rate.

Interpolates joint/gripper positions and Cartesian poses when requested, holds
other commands, and marks steps that reuse a source step with a filled flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from oopsie_backend.training.convert_actions_states import decode_poses, encode_poses

INTERPOLATED = {"joint_position": "linear", "gripper_position": "linear", "cartesian_position": "pose"}


@dataclass
class Plan:
    rate: float
    index: np.ndarray  # source step at or before each output step
    frac: np.ndarray  # position between ``index`` and the next source step, 0 where values are held
    filled: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    def head(self, n: int) -> Plan:
        return Plan(self.rate, self.index[:n], self.frac[:n], self.filled[:n])


def plan(length: int, source_rate: float, rate: float | None, upsample: str) -> Plan:
    rate = float(rate or source_rate)
    if rate > source_rate and upsample == "native":
        rate = float(source_rate)
    n = max(1, int(np.floor(length * rate / source_rate + 1e-9)))
    position = np.arange(n) * source_rate / rate
    index = np.minimum(np.floor(position + 1e-9).astype(np.int64), length - 1)
    frac = position - index if rate > source_rate and upsample == "interpolate" else np.zeros(n)
    frac = np.where(index == length - 1, 0.0, np.clip(frac, 0.0, 1.0))
    filled = np.concatenate([[False], index[1:] == index[:-1]])
    return Plan(rate, index, frac, filled)


def resample(key: str, value: np.ndarray, p: Plan, orientation: str, arms: int) -> np.ndarray:
    kind = INTERPOLATED.get(key)
    if kind is None or not p.frac.any():
        return value[p.index]
    nxt = np.minimum(p.index + 1, len(value) - 1)
    f = p.frac[:, None]
    if kind == "linear":
        return value[p.index] * (1 - f) + value[nxt] * f
    poses = []
    for (p0, r0), (p1, r1) in zip(decode_poses(value[p.index], arms, orientation), decode_poses(value[nxt], arms, orientation)):
        step = Rotation.from_rotvec((r1 * r0.inv()).as_rotvec() * f)
        poses.append((p0 * (1 - f) + p1 * f, step * r0))
    return encode_poses(poses, orientation)
