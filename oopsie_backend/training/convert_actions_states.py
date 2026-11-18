"""Convert action and state arrays to a recipe's target representation.

Derives compatible position/velocity commands, converts pose orientations,
normalizes grippers and optionally generates actions from consecutive states.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from oopsie_backend.training.recipe import STATE_KEYS, STATE_SOURCE
from oopsie_backend.training.read_episode import Skip, SourceEpisode

ROTATION_WIDTH = {"quat": 4, "matrix": 9, "rot6d": 6, "rotvec": 3}


def rotation_width(rep: str) -> int:
    return 3 if rep.startswith("euler_") else ROTATION_WIDTH[rep]


def _decode_rotation(block: np.ndarray, rep: str) -> Rotation:
    if rep == "quat":
        return Rotation.from_quat(block)
    if rep == "matrix":
        return Rotation.from_matrix(block.reshape(-1, 3, 3))
    if rep == "rot6d":
        u1 = block[:, :3] / np.linalg.norm(block[:, :3], axis=1, keepdims=True)
        u2 = block[:, 3:] - np.sum(block[:, 3:] * u1, axis=1, keepdims=True) * u1
        u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
        return Rotation.from_matrix(np.stack([u1, u2, np.cross(u1, u2)], axis=2))
    if rep == "rotvec":
        return Rotation.from_rotvec(block)
    if rep.startswith("euler_"):
        return Rotation.from_euler(rep[len("euler_"):], block)
    raise Skip(f"unknown orientation representation {rep!r}")


def _encode_rotation(rot: Rotation, rep: str) -> np.ndarray:
    if rep == "quat":
        return rot.as_quat()
    if rep == "matrix":
        return rot.as_matrix().reshape(-1, 9)
    if rep == "rot6d":
        m = rot.as_matrix()
        return np.concatenate([m[:, :, 0], m[:, :, 1]], axis=1)
    if rep == "rotvec":
        return rot.as_rotvec()
    return rot.as_euler(rep[len("euler_"):])


def decode_poses(array: np.ndarray, arms: int, rep: str | None) -> list[tuple[np.ndarray, Rotation]]:
    """``(position, rotation)`` per arm of a ``(T, D)`` pose array."""
    width = array.shape[1] // arms
    if array.shape[1] % arms:
        raise Skip(f"pose width {array.shape[1]} does not split into {arms} arms")
    if width != 7:
        if not rep:
            raise Skip(f"pose width {width} without an orientation representation")
        if width != 3 + rotation_width(rep):
            raise Skip(f"pose width {width} does not match {rep}")
    rep = "quat" if width == 7 else rep
    return [
        (array[:, a * width : a * width + 3], _decode_rotation(array[:, a * width + 3 : (a + 1) * width], rep))
        for a in range(arms)
    ]


def encode_poses(poses: list[tuple[np.ndarray, Rotation]], rep: str) -> np.ndarray:
    return np.concatenate([np.concatenate([p, _encode_rotation(r, rep)], axis=1) for p, r in poses], axis=1)


def _velocity_from_poses(target, state, rate: float) -> np.ndarray:
    return np.concatenate(
        [np.concatenate([(pt - ps) * rate, (rt * rs.inv()).as_rotvec() * rate], axis=1) for (pt, rt), (ps, rs) in zip(target, state)],
        axis=1,
    )


def _poses_from_velocity(velocity: np.ndarray, state, rate: float) -> list[tuple[np.ndarray, Rotation]]:
    width = velocity.shape[1] // len(state)
    if width != 6:
        raise Skip(f"cartesian_velocity width {width} per arm is not xyz plus a rotation vector")
    out = []
    for a, (ps, rs) in enumerate(state):
        v = velocity[:, a * 6 : (a + 1) * 6]
        out.append((ps + v[:, :3] / rate, Rotation.from_rotvec(v[:, 3:] / rate) * rs))
    return out


class Unifier:
    """Converts the arrays of one source episode under a recipe's ``actions`` and ``state`` settings."""

    def __init__(self, recipe: dict, gripper_range: tuple[float, float] | None):
        self.actions_cfg, self.state_cfg = recipe["actions"], recipe["state"]
        self.orientation = self.actions_cfg["orientation"]
        self.gripper_range = gripper_range

    def _gripper(self, value: np.ndarray) -> np.ndarray:
        if self.actions_cfg["gripper"] == "raw":
            return value
        if self.gripper_range is None:
            raise Skip("no gripper range for normalization")
        lo, hi = self.gripper_range
        return np.clip((value - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(value)

    def _state_poses(self, ep: SourceEpisode):
        if "cartesian_position" not in ep.states:
            raise Skip("no cartesian_position state")
        return decode_poses(ep.states["cartesian_position"], ep.arms, ep.profile.get("robot_state_orientation_representation"))

    def _action_poses(self, ep: SourceEpisode):
        return decode_poses(ep.actions["cartesian_position"], ep.arms, ep.profile.get("orientation_representation"))

    def action(self, key: str, ep: SourceEpisode, derive: bool) -> np.ndarray:
        a, s, rate = ep.actions, ep.states, ep.rate
        if key == "cartesian_position" and key in a:
            return encode_poses(self._action_poses(ep), self.orientation)
        if key == "gripper_position" and key in a:
            return self._gripper(a[key])
        if key in a:
            return a[key]
        if not derive:
            raise Skip(f"no {key} action")
        if key == "joint_position" and "joint_velocity" in a and "joint_position" in s:
            return s["joint_position"] + a["joint_velocity"] / rate
        if key == "joint_velocity" and "joint_position" in a and "joint_position" in s:
            return (a["joint_position"] - s["joint_position"]) * rate
        if key == "cartesian_position" and "cartesian_velocity" in a:
            return encode_poses(_poses_from_velocity(a["cartesian_velocity"], self._state_poses(ep), rate), self.orientation)
        if key == "cartesian_velocity" and "cartesian_position" in a:
            return _velocity_from_poses(self._action_poses(ep), self._state_poses(ep), rate)
        if key == "gripper_position" and "gripper_binary" in a:
            return a["gripper_binary"]
        if key == "gripper_binary" and "gripper_position" in a:
            return (self._gripper(a["gripper_position"]) > 0.5).astype(np.float64)
        raise Skip(f"no {key} action and no way to derive it from {sorted(a)}")

    def state(self, key: str, ep: SourceEpisode) -> np.ndarray:
        if key not in ep.states:
            raise Skip(f"no {key} state")
        if key == "cartesian_position":
            return encode_poses(self._state_poses(ep), self.orientation)
        if key == "gripper_position":
            return self._gripper(ep.states[key])
        return ep.states[key]

    def _from_states(self, key: str, ep: SourceEpisode) -> bool:
        """Whether the action ``key`` is generated from the states."""
        mode = self.actions_cfg["from_states"]
        if mode == "always":
            return True
        if mode == "never":
            return False
        try:
            self.action(key, ep, derive=True)
            return False
        except Skip:
            if key in STATE_SOURCE and STATE_SOURCE[key] in ep.states:
                return True
            raise

    def unify(self, ep: SourceEpisode) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """``(actions, states, bases)`` by key, at the source rate.

        ``bases`` holds, for every action generated from the states, the state it
        is generated from (see ``action_from_state``).
        """
        bases = {}
        if self.actions_cfg["mode"] == "target":
            action_keys, state_keys, derive = self.actions_cfg["keys"], self.state_cfg["keys"], True
            bases = {k: self.state(STATE_SOURCE[k], ep) for k in action_keys if self._from_states(k, ep)}
            action_keys = [k for k in action_keys if k not in bases]
        else:
            declared = ep.profile.get("action_space") or sorted(ep.actions)
            action_keys = [k for k in declared if k in ep.actions]
            state_keys = [k for k in STATE_KEYS if k in ep.states]
            derive = False
        actions = {k: self.action(k, ep, derive) for k in action_keys}
        states = {k: self.state(k, ep) for k in state_keys}
        for part, arrays in (("action", actions), ("state", states), ("state", bases)):
            for k, v in arrays.items():
                if k.startswith("gripper_") and v.shape[1] != ep.arms:
                    raise Skip(f"{k} {part} has width {v.shape[1]}, expected one per arm ({ep.arms})")
        return actions, states, bases


def action_from_state(key: str, state: np.ndarray, rate: float, orientation: str, arms: int) -> np.ndarray:
    """The action ``key`` of every step but the last, from the resampled state it is generated from."""
    if key in ("joint_position", "cartesian_position", "gripper_position"):
        return state[1:]
    if key == "gripper_binary":
        return (state[1:] > 0.5).astype(np.float64)
    if key == "joint_velocity":
        return (state[1:] - state[:-1]) * rate
    poses = decode_poses(state, arms, orientation)
    return _velocity_from_poses(
        [(p[1:], r[1:]) for p, r in poses], [(p[:-1], r[:-1]) for p, r in poses], rate
    )
