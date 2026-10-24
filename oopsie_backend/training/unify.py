"""Write the unified episodes of a recipe: ``episodes/<episode_key>.h5`` and ``episodes/<episode_key>/<role>.mp4``.

A unified episode holds ``action/<key>`` and ``state/<key>`` as ``(T, D)`` float32
at the recipe rate, each at its own width (writers pad to a common width), and
the attrs ``fps``, ``source_fps``, ``unify_hash``, ``gripper_range``, the source
file's ``source_size`` and ``source_mtime_ns``, and ``camera/<role>`` (the video
file, or empty for a blank role). It is rewritten only when one of these
changed. Labels are not stored here but in the manifest, so new annotations
never require a rewrite.

Rate changes take, for each output step, the latest source step at or before
it; actions are converted at the source rate first.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.training.cameras import assign_roles, transcode
from oopsie_backend.training.convert import Unifier
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


def output_steps(length: int, source_rate: float, rate: float) -> np.ndarray:
    n = max(1, int(np.floor(length * rate / source_rate + 1e-9)))
    return np.minimum(np.floor(np.arange(n) * source_rate / rate + 1e-9).astype(np.int64), length - 1)


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
        }


def unify_episode(job: tuple) -> dict:
    """Unify one episode; returns its manifest fields (``status``, ``reason`` and the ``_summary``)."""
    key, repo_path, source, out_dir, recipe, uhash, gripper_range = job
    out = out_dir / f"{key}.h5"
    st = source.stat()
    expected = {
        "unify_hash": uhash,
        "source_size": st.st_size,
        "source_mtime_ns": st.st_mtime_ns,
        "gripper_range": json.dumps(gripper_range),
    }
    if _current(out, expected):
        return {"status": "reused", "reason": None, **_summary(out)}
    try:
        ep = read_source(source, repo_path)
        actions, states = Unifier(recipe, gripper_range).unify(ep)
        rate = float(recipe["rate_hz"] or ep.rate)
        steps = output_steps(ep.length, ep.rate, rate)
        roles = assign_roles(sorted(ep.videos), recipe["cameras"])
        cams = recipe["cameras"]
        videos = {}
        for role, camera in roles.items():
            if camera is None:
                videos[role] = ""
                continue
            rel = f"{key}/{role}.mp4"
            transcode(ep.videos[camera], ep.length, steps, out_dir / rel, tuple(cams["image_size"]), rate, cams["crf"])
            videos[role] = rel
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        with h5py.File(tmp, "w") as f:
            for name, arrays in (("action", actions), ("state", states)):
                group = f.create_group(name)
                for k, v in arrays.items():
                    group.create_dataset(k, data=v[steps].astype(np.float32))
            f.attrs.update(expected)
            f.attrs.update({"fps": rate, "source_fps": ep.rate, "num_steps": len(steps)})
            f.attrs.update({f"camera/{r}": v for r, v in videos.items()})
            f.attrs["camera_sources"] = json.dumps(roles)
        os.replace(tmp, out)
        return {"status": "written", "reason": None, **_summary(out)}
    except Skip as exc:
        return {"status": "skipped", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the materialization
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
