"""Action features read from the episode files, and the statistics built on them (entries 36 and 38 to 41).

Features are computed from the joint and gripper states, which every episode records,
so they do not depend on the action space. Speeds and idle times are relative to the
episode's own 95th percentile joint speed, and the smoothness is the dimensionless
log jerk, so episodes of different robots are comparable.

Per episode:
- ``mean_speed``: mean joint speed (rad/s);
- ``idle_share``: share of steps below ``idle_speed_fraction`` of the 95th percentile speed;
- ``path_efficiency``: joint-space distance from start to end over the distance travelled;
- ``log_jerk``: log dimensionless jerk of the joint speed profile (higher is smoother);
- ``oscillation``: joint velocity sign changes per joint and second, ignoring idle motion;
- ``sudden_stops``: falls from above half the 95th percentile speed to idle within
  ``sudden_stop_seconds``, per minute;
- ``gripper_events``: gripper changes between the ``gripper_low`` and ``gripper_high``
  levels of the robot's normalized range, per minute;
- ``regrasp`` and ``closed_on_nothing``: two or more closes, and a close that reaches
  ``gripper_full``; only for robots whose closed end is set in ``gripper_closed``;
- ``tracking_gap``: distance between a joint position command and the next joint state,
  over the mean step; only for episodes commanding joint positions.

Extraction reads every episode file twice, first for the robots' gripper ranges.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from oopsie_backend.common.pools import process_pool
from oopsie_backend.common.repos import episode_path
from oopsie_backend.index.store import KEY
from oopsie_backend.stats.analysis.common import (
    FAILURE,
    POLICY_CLASSES,
    SUCCESS_OUTCOMES,
    TELEOP,
    UNCATEGORIZED,
    distribution,
)
from oopsie_backend.training.unify import gripper_extent, robot_id

FEATURES = [
    "mean_speed", "idle_share", "path_efficiency", "log_jerk", "oscillation", "sudden_stops",
    "gripper_events", "regrasp", "closed_on_nothing", "tracking_gap",
]
SIGNATURES = ["closed_on_nothing", "regrasp", "sudden_stops", "idle_share", "oscillation"]
FEATURE_SCHEMA = {KEY: pl.String, **{f: pl.Float64 for f in FEATURES}}


def _array(ds) -> np.ndarray | None:
    """A ``(T, D)`` float array of a dataset; one column for a ``(T,)`` dataset; ``None`` when empty."""
    if not isinstance(ds, h5py.Dataset) or not ds.shape:
        return None
    a = np.asarray(ds, dtype=np.float64)
    return a.reshape(len(a), -1)


def _read(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        q = _array(f["observations/robot_states/joint_position"])
        g = _array(f.get("observations/robot_states/gripper_position"))
        a = _array(f.get("actions/joint_position"))
    if q is None or q.shape[1] == 0:
        raise KeyError("no joint_position state")
    return q, g, a


def _transitions(level: np.ndarray, low: float, high: float) -> list[tuple[int, bool]]:
    """``(step, to_high)`` of every change between the low and high hysteresis states."""
    state, out = None, []
    for i, v in enumerate(level):
        new = True if v >= high else False if v <= low else state
        if state is not None and new != state:
            out.append((i, new))
        state = new
    return out


def motion_features(
    q: np.ndarray,
    g: np.ndarray | None,
    a: np.ndarray | None,
    rate: float,
    gripper_range: tuple[float, float] | None,
    closed: str | None,
    cfg: dict,
) -> dict:
    """Features of one episode, or of a slice of one."""
    out = dict.fromkeys(FEATURES, math.nan)
    if len(q) < 4 or rate <= 0:
        return out
    steps = np.diff(q, axis=0)
    speed = np.linalg.norm(steps, axis=1) * rate
    duration_min = len(q) / rate / 60
    p95 = float(np.quantile(speed, 0.95))
    idle = cfg["idle_speed_fraction"] * p95
    travelled = float(np.linalg.norm(steps, axis=1).sum())
    out["mean_speed"] = float(speed.mean())
    out["idle_share"] = float((speed < idle).mean()) if p95 > 0 else 1.0
    out["path_efficiency"] = float(np.linalg.norm(q[-1] - q[0]) / travelled) if travelled > 0 else math.nan
    jerk = np.diff(speed, n=2) * rate * rate
    integral = float((jerk**2).sum() / rate)
    if speed.max() > 0 and integral > 0:
        dur = len(speed) / rate
        out["log_jerk"] = -math.log(dur**3 / speed.max() ** 2 * integral)
    velocity = steps * rate
    moving = np.abs(velocity) > cfg["idle_speed_fraction"] * np.quantile(np.abs(velocity), 0.95, axis=0)
    signs = np.sign(velocity) * moving
    flips = ((signs[1:] * signs[:-1]) < 0).sum()
    out["oscillation"] = float(flips / q.shape[1] / (len(q) / rate))
    if p95 > 0:
        window = max(1, math.ceil(cfg["sudden_stop_seconds"] * rate))
        onsets = np.nonzero((speed[1:] < idle) & (speed[:-1] >= idle))[0] + 1
        stops = sum(speed[max(0, i - window) : i].max() > 0.5 * p95 for i in onsets)
        out["sudden_stops"] = stops / duration_min
    if g is not None and gripper_range is not None and gripper_range[1] > gripper_range[0]:
        lo, hi = gripper_range
        level = np.clip((g - lo) / (hi - lo), 0, 1)
        events, closes, empty = 0, 0, False
        for col in level.T:
            c = col if closed != "low" else 1 - col
            trans = _transitions(c, cfg["gripper_low"], cfg["gripper_high"])
            events += len(trans)
            if closed:
                for k, (i, to_high) in enumerate(trans):
                    if to_high:
                        closes += 1
                        end = trans[k + 1][0] if k + 1 < len(trans) else len(c)
                        empty |= bool(c[i:end].max() >= cfg["gripper_full"])
        out["gripper_events"] = events / duration_min
        if closed:
            out["regrasp"] = float(closes >= 2)
            out["closed_on_nothing"] = float(empty)
    if a is not None and a.shape == q.shape:
        gap = np.linalg.norm(a[:-1] - q[1:], axis=1).mean()
        mean_step = np.linalg.norm(steps, axis=1).mean()
        out["tracking_gap"] = float(gap / mean_step) if mean_step > 0 else math.nan
    return out


def _extent_job(path: Path) -> tuple[float, float] | None:
    try:
        return gripper_extent(path)
    except (OSError, KeyError):
        return None


def gripper_ranges(paths: dict[str, Path], robots: dict[str, str], workers: int) -> dict[str, tuple[float, float]]:
    """Per robot, the 5th percentile of the episodes' gripper minima and the 95th of their maxima."""
    keys = list(paths)
    with process_pool(workers) as pool:
        extents = list(pool.map(_extent_job, [paths[k] for k in keys], chunksize=64))
    per_robot = defaultdict(list)
    for key, extent in zip(keys, extents):
        if extent is not None:
            per_robot[robots[key]].append(extent)
    return {
        r: (float(np.quantile([e[0] for e in ex], 0.05)), float(np.quantile([e[1] for e in ex], 0.95)))
        for r, ex in per_robot.items()
    }


def _episode_job(job: tuple) -> list[dict]:
    key, path, rate, gripper_range, closed, cfg, windows = job
    try:
        q, g, a = _read(path)
    except (OSError, KeyError):
        return []
    rows = [{KEY: key, "window": "episode", "segment": None, **motion_features(q, g, a, rate, gripper_range, closed, cfg)}]
    w = max(4, round(cfg["window_seconds"] * rate))
    for n, start in enumerate(windows):
        i = round(start * rate)
        for name, lo, hi in (("before", i - w, i), ("after", i, i + w)):
            lo, hi = max(0, lo), min(len(q), hi)
            sl = slice(lo, hi)
            feats = motion_features(
                q[sl], g[sl] if g is not None else None, a[sl] if a is not None else None, rate, gripper_range, closed, cfg
            )
            rows.append({KEY: key, "window": name, "segment": n, **feats})
    return rows


def extract(
    episodes: pl.DataFrame,
    data_dir: Path,
    cfg: dict,
    workers: int = 8,
    segments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Features of every episode (``window == "episode"``), and around every segment start of ``segments``."""
    dcfg = cfg["data"]
    rows = episodes.select(KEY, "dataset_repo", "repo_path", "control_freq", "robot_name", "gripper_name").to_dicts()
    paths = {r[KEY]: episode_path(data_dir, r["dataset_repo"], r["repo_path"]) for r in rows}
    robots = {r[KEY]: robot_id(r) for r in rows}
    ranges = gripper_ranges(paths, robots, workers)
    closed = dcfg.get("gripper_closed") or {}
    starts = defaultdict(list)
    if segments is not None:
        for key, start in segments.select(KEY, "start").iter_rows():
            starts[key].append(start)
    jobs = [
        (r[KEY], paths[r[KEY]], float(r["control_freq"] or 0), ranges.get(robots[r[KEY]]), closed.get(robots[r[KEY]]),
         dcfg, starts.get(r[KEY], []))
        for r in rows
    ]
    with process_pool(workers) as pool:
        out = [row for result in pool.map(_episode_job, jobs, chunksize=16) for row in result]
    schema = {KEY: pl.String, "window": pl.String, "segment": pl.Int64, **{f: pl.Float64 for f in FEATURES}}
    return pl.DataFrame(out, schema=schema).with_columns(pl.col(FEATURES).fill_nan(None))


def _summary(df: pl.DataFrame, features: list[str]) -> dict:
    return {"n": df.height, **{f: distribution(df[f]) for f in features}}


def teleop_vs_policy_motion(feats: pl.DataFrame, episodes: pl.DataFrame) -> dict:
    """Entry 36: feature distributions of teleoperation against policy episodes."""
    df = feats.join(episodes.select(KEY, "policy_class"), on=KEY)
    return {
        "teleop": _summary(df.filter(pl.col("policy_class") == TELEOP), FEATURES),
        "policy": _summary(df.filter(pl.col("policy_class").is_in(POLICY_CLASSES)), FEATURES),
    }


def signatures_by_category(feats: pl.DataFrame, labels: pl.DataFrame, min_n: int) -> dict:
    """Entry 39: signature features per failure category, against successful episodes."""
    df = feats.join(labels.select(KEY, "outcome", "categories"), on=KEY)
    out = {"success": _summary(df.filter(pl.col("outcome").is_in(SUCCESS_OUTCOMES)), SIGNATURES)}
    exploded = df.filter(pl.col("outcome") == FAILURE).with_columns(
        pl.when(pl.col("categories").list.len() > 0).then(pl.col("categories")).otherwise(pl.lit([UNCATEGORIZED]))
    ).explode("categories")
    for (cat,), g in exploded.group_by("categories"):
        if g.height >= min_n:
            out[str(cat)] = _summary(g, SIGNATURES)
    return out


def failed_vs_successful(feats: pl.DataFrame, labels: pl.DataFrame, episodes: pl.DataFrame, min_n: int) -> dict:
    """Entry 40: feature medians of failed and successful episodes within each policy class and lab."""
    df = feats.join(labels.select(KEY, "outcome"), on=KEY).join(
        episodes.select(KEY, "policy_class", "lab_id"), on=KEY
    ).with_columns(
        pl.when(pl.col("outcome") == FAILURE).then(pl.lit("failed"))
        .when(pl.col("outcome").is_in(SUCCESS_OUTCOMES)).then(pl.lit("successful")).alias("group")
    ).filter(pl.col("group").is_not_null())
    out = {}
    for column in ("policy_class", "lab_id"):
        per = {}
        for (value,), g in df.group_by(column):
            sizes = dict(g.group_by("group").len().iter_rows())
            if sizes.get("failed", 0) < min_n or sizes.get("successful", 0) < min_n:
                continue
            medians = g.group_by("group").agg([pl.col(f).median() for f in FEATURES])
            m = {row["group"]: row for row in medians.to_dicts()}
            per[str(value)] = {
                "sizes": sizes,
                **{f: {"failed": m["failed"][f], "successful": m["successful"][f]} for f in FEATURES},
            }
        out[column] = per
    return out


def failure_prediction(feats: pl.DataFrame, labels: pl.DataFrame, episodes: pl.DataFrame, folds: int = 5) -> dict:
    """Entry 41: AUROC of a logistic regression on the features, cross-validated with labs held out."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = feats.join(labels.select(KEY, "outcome"), on=KEY).join(episodes.select(KEY, "lab_id"), on=KEY).filter(
        (pl.col("outcome") == FAILURE) | pl.col("outcome").is_in(SUCCESS_OUTCOMES)
    )
    usable = [f for f in FEATURES if df[f].is_not_null().any() and df[f].is_not_nan().any()]
    x = df.select(usable).to_numpy()
    y = (df["outcome"] == FAILURE).to_numpy().astype(int)
    groups = df["lab_id"].fill_null("").to_numpy()
    if len(set(y)) < 2 or len(set(groups)) < folds:
        return {"available": False, "reason": "too few labs or a single outcome"}
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    scores = []
    for train, test in GroupKFold(folds).split(x, y, groups):
        if len(set(y[test])) < 2:
            continue
        model.fit(x[train], y[train])
        scores.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    model.fit(x, y)
    return {
        "available": True,
        "episodes": len(y),
        "features": usable,
        "auroc_per_fold": scores,
        "auroc_mean": float(np.mean(scores)) if scores else None,
        "coefficients": dict(zip(usable, model[-1].coef_[0].tolist())),
    }


def signatures_before_failure(windows: pl.DataFrame) -> dict:
    """Entry 38: features before and after each segment start, as differences to the whole episode."""
    base = windows.filter(pl.col("window") == "episode").drop("window", "segment")
    out = {}
    for name in ("before", "after"):
        w = windows.filter(pl.col("window") == name).join(base, on=KEY, suffix="_episode")
        out[name] = {f: distribution((w[f] - w[f"{f}_episode"]).to_list()) for f in FEATURES}
    return out


def compute(feats: pl.DataFrame, episodes: pl.DataFrame, labels: pl.DataFrame, cfg: dict) -> dict:
    """Every data-based entry that needs no segments, keyed by entry; ``feats`` holds the episode rows of ``extract``."""
    min_n = cfg["min_group_episodes"]
    feats = feats.filter(pl.col("window") == "episode").drop("window", "segment")
    return {
        "36_teleop_vs_policy_motion": teleop_vs_policy_motion(feats, episodes),
        "39_signatures_by_category": signatures_by_category(feats, labels, min_n),
        "40_failed_vs_successful": failed_vs_successful(feats, labels, episodes, min_n),
        "41_failure_prediction": failure_prediction(feats, labels, episodes),
    }
