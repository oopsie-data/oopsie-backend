"""Write the unified episodes of a recipe: ``episodes/<episode_key>.h5`` and ``episodes/<episode_key>/<role>.mp4``.

A unified episode holds ``action/<key>`` and ``state/<key>`` as ``(T, D)`` float32
at the recipe rate, each at its own width (writers pad to a common width),
``filled`` (``(T,)`` bool, the steps that add no new source step), and the attrs
``fps``, ``source_fps``, ``unify_hash``, ``gripper_range``, ``unified_cameras``, the source file's
``source_size`` and ``source_mtime_ns``, ``actions_from_states`` (the action keys
generated from the states) and ``camera/<role>`` (the video file, or empty for a
blank role). It is rewritten only when one of these
changed. Labels are not stored here but in the manifest, so new annotations
never require a rewrite.

Actions and states are converted at the source rate, then resampled
(``training/resample.py``); actions generated from the states are computed after
resampling. Videos take the frame of the latest source step at or before each step.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.training.cameras import assign_roles, transcode
from oopsie_backend.training.convert import Unifier, action_from_state
from oopsie_backend.training.recipe import STATE_SOURCE
from oopsie_backend.training.resample import plan, resample
from oopsie_backend.training.source import Skip, read_source

ROBOT = ("robot_name", "gripper_name")


def robot_id(row: dict) -> str:
    return "/".join(str(row.get(k) or "") for k in ROBOT)


def gripper_extent(path: Path) -> tuple[float, float] | None:
    """Min and max of the episode's gripper position state and action."""
    values = []
    with h5py.File(path, "r") as f:
        for name in ("observations/robot_states/gripper_position", "actions/gripper_position"):
            ds = f.get(name)
            if isinstance(ds, h5py.Dataset) and ds.shape is not None and ds.size:
                values.append(np.asarray(ds, dtype=np.float64).ravel())
    if not values:
        return None
    v = np.concatenate(values)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if v.size else None


def gripper_job(args: tuple[str, Path]) -> tuple[str, tuple[float, float] | None]:
    robot, path = args
    try:
        return robot, gripper_extent(path)
    except OSError:
        return robot, None


def _current(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            return all(f.attrs.get(k) == v for k, v in expected.items())
    except OSError:
        return False


def _widths(group: h5py.Group) -> dict[str, int]:
    return {k: int(group[k].shape[1]) for k in group}


def _summary(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "num_steps": int(f.attrs["num_steps"]),
            "fps": float(f.attrs["fps"]),
            "action_widths": json.dumps(_widths(f["action"])),
            "state_widths": json.dumps(_widths(f["state"])),
            "cameras": json.dumps({k[len("camera/"):]: str(f.attrs[k]) for k in f.attrs if k.startswith("camera/")}),
            "actions_from_states": str(f.attrs["actions_from_states"]),
            "filled_steps": int(np.asarray(f["filled"]).sum()),
        }


def unify_episode(job: tuple) -> dict:
    """Unify one episode; returns its manifest fields (``status``, ``reason`` and the ``_summary``)."""
    key, repo_path, source, out_dir, recipe, uhash, gripper_range, unified = job
    out = out_dir / f"{key}.h5"
    st = source.stat()
    expected = {
        "unify_hash": uhash,
        "source_size": st.st_size,
        "source_mtime_ns": st.st_mtime_ns,
        "gripper_range": json.dumps(gripper_range),
        "unified_cameras": json.dumps(sorted((e["camera"], e["name"]) for e in unified or [])),
    }
    if _current(out, expected):
        return {"status": "reused", "reason": None, **_summary(out)}
    try:
        ep = read_source(source, repo_path)
        actions, states, bases = Unifier(recipe, gripper_range).unify(ep)
        p = plan(ep.length, ep.rate, recipe["rate_hz"], recipe["upsample"])
        orientation = recipe["actions"]["orientation"]
        actions = {k: resample(k, v, p, orientation, ep.arms) for k, v in actions.items()}
        states = {k: resample(k, v, p, orientation, ep.arms) for k, v in states.items()}
        if bases:
            if len(p) < 2:
                raise Skip("too short to generate actions from the states")
            for k, v in bases.items():
                actions[k] = action_from_state(k, resample(STATE_SOURCE[k], v, p, orientation, ep.arms), p.rate, orientation, ep.arms)
            p = p.head(len(p) - 1)
            actions = {k: v[: len(p)] for k, v in actions.items()}
            states = {k: v[: len(p)] for k, v in states.items()}
        actions = {k: actions[k] for k in (recipe["actions"]["keys"] or actions)}
        rate = p.rate
        if unified is None:
            raise Skip("no unified camera names; run oopsie index camera-names")
        names = {e["name"]: e["camera"] for e in unified if e["name"] and e["camera"] in ep.videos}
        if not names:
            raise Skip("no camera has a unified name")
        roles = assign_roles(names, recipe["cameras"])
        cams = recipe["cameras"]
        videos = {}
        for role, camera in roles.items():
            if camera is None:
                videos[role] = ""
                continue
            rel = f"{key}/{role}.mp4"
            size = tuple(cams["image_size"]) if cams["image_size"] else None
            transcode(ep.videos[camera], ep.length, p.index, out_dir / rel, size, rate, cams["crf"])
            videos[role] = rel
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        with h5py.File(tmp, "w") as f:
            for name, arrays in (("action", actions), ("state", states)):
                group = f.create_group(name)
                for k, v in arrays.items():
                    group.create_dataset(k, data=v.astype(np.float32))
            f.create_dataset("filled", data=p.filled)
            f.attrs.update(expected)
            f.attrs.update({"fps": rate, "source_fps": ep.rate, "num_steps": len(p), "actions_from_states": json.dumps(sorted(bases))})
            f.attrs.update({f"camera/{r}": v for r, v in videos.items()})
            f.attrs["camera_sources"] = json.dumps(roles)
        os.replace(tmp, out)
        return {"status": "written", "reason": None, **_summary(out)}
    except Skip as exc:
        return {"status": "skipped", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the materialization
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
