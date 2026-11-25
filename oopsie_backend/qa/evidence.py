"""Evidence for one flagged episode and check, for the evidence panel of the action QA
report (served by ``oopsie_backend/qa/serve_report.py``).

Read on demand from the episode file, so the report itself carries no evidence data:
the episode's camera videos (with the moment to seek to), small charts of the values the
check looked at, and plain values for checks with nothing to plot (names, shapes).
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.common.h5 import is_h5, strip_h5_suffix
from oopsie_backend.common.videos import episode_videos
from oopsie_backend.qa.action_qa import (
    STATE_KEY_FOR_ACTION,
    _is_gripper_joint,
    _joint_pairing,
    _state_rotation_to_quat,
    load_action_state_arrays,
    profile_control_freq,
)
from oopsie_backend.qa.video_checks import (
    _continuous_quaternions,
    drop_ignored_cameras,
    state_step,
)

HF_ORG = "OopsieData-Submissions"
WINDOW_S = 3.0  # seconds shown on each side of a flagged moment
MAX_POINTS = 600  # points per series, so a long episode stays light
_AT_TIME = re.compile(r"at ([\d.]+)s")


def episode_path(data_dir: Path, dataset_repo: str, repo_path: str) -> Path:
    """The episode's HDF5 file under ``data_dir`` (the org folder or its parent); refuses paths
    that would leave it."""
    data_dir = data_dir.resolve()
    name = dataset_repo.split("/", 1)[-1]
    base = data_dir / name if data_dir.name == HF_ORG else data_dir / dataset_repo
    path = (base / repo_path).resolve()
    if not path.is_relative_to(data_dir) or not is_h5(path.name):
        raise ValueError(f"not an episode under the data dir: {dataset_repo}/{repo_path}")
    return path


def _chart(title: str, unit: str, t: np.ndarray, series: list[tuple[str, np.ndarray]],
           markers: list[float], window: tuple[float, float] | None = None, hline: float | None = None) -> dict:
    """A line chart: series over time ``t`` (s), optionally cut to ``window`` and thinned to
    MAX_POINTS, with vertical markers at the flagged moments and an optional limit line."""
    keep = np.ones(len(t), dtype=bool) if window is None else (t >= window[0]) & (t <= window[1])
    idx = np.flatnonzero(keep)
    if len(idx) > MAX_POINTS:
        idx = idx[np.linspace(0, len(idx) - 1, MAX_POINTS).astype(int)]
    out = []
    for name, y in series:
        y = np.asarray(y, dtype=float)
        n = min(len(y), len(t))
        sel = idx[idx < n]
        out.append({"name": name, "x": np.round(t[sel], 3).tolist(),
                    "y": [None if not np.isfinite(v) else round(float(v), 5) for v in y[sel]]})
    return {"title": title, "unit": unit, "series": out, "markers": [round(m, 3) for m in markers],
            "hline": hline}


def _moments(findings: list[dict], hz: float | None) -> list[float]:
    """Flagged moments (s) from the findings: an explicit time_s, an "at Xs" in the text,
    or a first_step in the per-joint / per-arm details."""
    times = []
    for f in findings:
        if f.get("time_s") is not None:
            times.append(float(f["time_s"]))
        elif f.get("issue") and (m := _AT_TIME.search(f["issue"])):
            times.append(float(m.group(1)))
        elif hz:
            for item in (f.get("joints") or []) + (f.get("arms") or []):
                if isinstance(item, dict) and item.get("first_step") is not None:
                    times.append((item["first_step"] + 1) / hz)
                    break
    return sorted(set(times))


def _window(times: list[float]) -> tuple[float, float] | None:
    return (times[0] - WINDOW_S, times[0] + WINDOW_S) if times else None


def _cols(names: list[str], width: int, want: str | None, n: int = 3) -> list[int]:
    """Columns to plot: the named joint if given, else the first ``n`` non-gripper ones."""
    if want and want in names:
        return [names.index(want)]
    fit = len(names) == width
    return [c for c in range(width) if not (fit and _is_gripper_joint(names[c]))][:n] or list(range(min(n, width)))


def _label(names: list[str], c: int) -> str:
    return names[c] if c < len(names) else f"col {c}"


def _worst(findings: list[dict], list_key: str, value_key: str) -> dict | None:
    items = [i for f in findings for i in (f.get(list_key) or []) if isinstance(i, dict)]
    return max(items, key=lambda i: i.get(value_key) or 0) if items else None


def _joint_charts(test, findings, actions, states, profile, t, times, velocity: bool) -> list[dict]:
    key = "joint_velocity" if velocity else "joint_position"
    a, s = actions.get(key), states.get("joint_position")
    if a is None or a.ndim != 2:
        return []
    names = [str(n) for n in profile.get("action_joint_names") or []]
    state_names = [str(n) for n in profile.get("robot_state_joint_names") or []]
    worst = (_worst(findings, "joints", "max_step_rad") or _worst(findings, "worst_joints", "median_error_rad")
             or _worst(findings, "joints", "max_speed"))
    cols = _cols(names, a.shape[1], worst.get("joint") if worst else None)
    pairing = dict(_joint_pairing(names, state_names, a.shape[1], s.shape[1]) or []) if s is not None and s.ndim == 2 else {}
    hz = 1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0
    window = _window(times) if test in ("step_jump",) else None
    charts = []
    for c in cols:
        series = []
        if velocity:
            series.append((f"{_label(names, c)} velocity x step time", a[:, c] / hz))
            if c in pairing:
                series.append((f"{_label(state_names, pairing[c])} change per step (state)", np.r_[np.diff(s[:, pairing[c]]), np.nan]))
            unit = "rad per step"
        else:
            series.append((f"{_label(names, c)} command", a[:, c]))
            if c in pairing:
                series.append((f"{_label(state_names, pairing[c])} state", s[:, pairing[c]]))
            unit = "joint value"
        charts.append(_chart(f"{_label(names, c)}: {'velocity vs real motion' if velocity else 'command vs state'}",
                             unit, t, series, times, window))
    return charts


def _arm_quats_and_xyz(pose: np.ndarray, n_arms: int):
    w = pose.shape[1] // n_arms
    return [pose[:, w * a : w * a + 3] for a in range(n_arms)], w


def _cartesian_charts(test, findings, actions, states, profile, t, times) -> list[dict]:
    a, s = actions.get("cartesian_position"), states.get("cartesian_position")
    if a is None or a.ndim != 2:
        return []
    n_arms = 2 if profile.get("is_biarm") else 1
    charts = []
    if test == "rotation_jump" and a.shape[1] == 7 * n_arms:
        worst = _worst(findings, "arms", "max_step_deg") or {"arm": 1}
        arm = int(worst.get("arm", 1)) - 1
        q = _continuous_quaternions(a)[:, 7 * arm + 3 : 7 * arm + 7]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))
        limit = next((f.get("limit_step_deg") for f in findings if f.get("limit_step_deg")), None)
        charts.append(_chart(f"arm {arm + 1}: commanded rotation per step", "degrees", t[1:], [("rotation per step", angle)],
                             times, _window(times), limit))
        return charts
    if test == "pose_tracking" and s is not None and a.shape[1] == 7 * n_arms:
        sw = s.shape[1] // n_arms
        for arm in range(n_arms):
            pos_err = np.linalg.norm(a[:-1, 7 * arm : 7 * arm + 3] - s[1:, sw * arm : sw * arm + 3], axis=1) * 100
            charts.append(_chart(f"arm {arm + 1}: commanded position vs where the robot ends up", "cm", t[:-1],
                                 [("position error", pos_err)], times))
            sq = _state_rotation_to_quat(s[:, sw * arm + 3 : sw * (arm + 1)], profile.get("robot_state_orientation_representation"))
            if sq is not None:
                aq = a[:, 7 * arm + 3 : 7 * arm + 7]
                aq = aq / np.linalg.norm(aq, axis=1, keepdims=True)
                ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(aq[:-1] * sq[1:], axis=1)), 0, 1)))
                charts.append(_chart(f"arm {arm + 1}: commanded orientation vs where the robot ends up", "degrees",
                                     t[:-1], [("orientation error", ang)], times))
        return charts
    # Units, deltas, arm order, copies: commanded xyz next to the state's.
    if s is None or s.ndim != 2:
        return []
    aw, sw = a.shape[1] // n_arms, s.shape[1] // n_arms
    for arm in range(n_arms):
        for axis, name in enumerate("xyz"):
            charts.append(_chart(f"arm {arm + 1} {name}: command vs state", "position", t,
                                 [(f"{name} command", a[:, aw * arm + axis]), (f"{name} state", s[:, sw * arm + axis])], times))
    return charts


def _cartesian_velocity_charts(actions, states, t, times) -> list[dict]:
    v, s = actions.get("cartesian_velocity"), states.get("cartesian_position")
    if v is None or v.ndim != 2 or v.shape[1] < 3:
        return []
    hz = 1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0
    series = [("commanded speed x step time", np.linalg.norm(v[:, :3], axis=1) / hz)]
    if s is not None and s.ndim == 2 and s.shape[1] >= 3:
        series.append(("end-effector distance moved per step (state)", np.r_[np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1), np.nan]))
    return [_chart("linear velocity vs real motion", "metres per step", t, series, times)]


def _action_vs_state_charts(key: str, actions, states, t, times) -> list[dict]:
    """Generic: the first columns of an action next to the state it's observed through."""
    keys = key.split("+") if "+" in key else [key]
    series = []
    for k in keys:
        if k in actions and actions[k].ndim >= 1:
            series.append((f"{k} (action)", actions[k].reshape(len(actions[k]), -1)[:, 0]))
    sk = STATE_KEY_FOR_ACTION.get(keys[0])
    if len(keys) == 1 and sk in states:
        series.append((f"{sk} (state)", states[sk].reshape(len(states[sk]), -1)[:, 0]))
    return [_chart(f"{key}: first column over time", "value", t, series, times)] if series else []


def _values(category: str, test: str, findings: list[dict], actions, states, profile) -> dict:
    """Plain values that explain format and metadata findings."""
    out: dict = {
        "action shapes": {k: list(v.shape) for k, v in actions.items()},
        "state shapes": {k: list(v.shape) for k, v in states.items()},
    }
    if category in ("joint_position", "joint_velocity") or test in ("required_state_keys",):
        out["action_joint_names"] = profile.get("action_joint_names")
        out["robot_state_joint_names"] = profile.get("robot_state_joint_names")
    if test == "placeholder_metadata":
        out.update({k: profile.get(k) for k in ("policy_name", "robot_name", "gripper_name")})
    out.update({k: profile.get(k) for k in ("action_space", "robot_state_keys", "control_freq", "is_biarm")})
    return out


def episode_evidence(data_dir: Path, dataset_repo: str, repo_path: str, category: str, test: str,
                     findings: list[dict], ignore_cameras=(), data_url_prefix: str = "data/") -> dict:
    """Everything the evidence panel shows for one episode and check."""
    path = episode_path(data_dir, dataset_repo, repo_path)
    with h5py.File(path, "r") as f:
        actions, states, profile = load_action_state_arrays(f)
    hz = profile_control_freq(profile)
    n = max([len(v) for v in list(actions.values()) + list(states.values()) if v.ndim >= 1] or [0])
    t = np.arange(n) / (hz or 1.0)
    times = _moments(findings, hz)

    videos, _ = drop_ignored_cameras(episode_videos(path, repo_path), ignore_cameras)
    root = Path(data_dir).resolve()
    flagged = {f.get("key") for f in findings}
    video_list = [
        {"cam": cam, "url": data_url_prefix + Path(p).resolve().relative_to(root).as_posix(), "flagged": cam in flagged}
        for cam, p in sorted(videos.items(), key=lambda kv: (kv[0] not in flagged, kv[0]))
        if Path(p).is_file() and Path(p).resolve().is_relative_to(root)
    ]

    charts: list[dict] = []
    try:
        if category == "joint_position":
            charts = _joint_charts(test, findings, actions, states, profile, t, times, velocity=False)
        elif category == "joint_velocity":
            charts = _joint_charts(test, findings, actions, states, profile, t, times, velocity=True)
        elif category == "cartesian_position":
            charts = _cartesian_charts(test, findings, actions, states, profile, t, times)
        elif category == "cartesian_velocity":
            charts = _cartesian_velocity_charts(actions, states, t, times)
        elif category == "video" and test in ("state_jump", "recording_gap", "periodic_skips", "choppy", "frozen"):
            step = state_step(states)
            if step is not None:
                window = _window(times) if test in ("state_jump", "recording_gap") else None
                charts = [_chart("robot motion per step (what the jump checks measure)", "scaled change",
                                 t[1:], [("motion", step)], times, window)]
        elif category == "general" and findings:
            key = str(findings[0].get("key", ""))
            if key in actions or "+" in key:  # an action key (or a duplicate pair "a+b")
                charts = _action_vs_state_charts(key, actions, states, t, times)
    except Exception as exc:  # evidence is a convenience: never fail the whole panel
        charts = [{"title": f"Couldn't draw a chart: {type(exc).__name__}: {exc}", "series": [], "markers": []}]

    return {
        "episode": strip_h5_suffix(repo_path),
        "control_freq": hz,
        "seek_s": times[0] if times else None,
        "moments_s": times,
        "videos": video_list,
        "charts": charts,
        "values": _values(category, test, findings, actions, states, profile),
    }
