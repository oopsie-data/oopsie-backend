"""Extract per-episode metadata from Oopsie HDF5 files and aggregate dashboard stats."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

_LS6_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_STATS_JSON = Path(
    os.environ.get("OOPSIE_STATS_OUT", _LS6_ROOT / "oopsie-docs/docs/assets/data/stats.json")
)

SCHEMA_VERSION = 2
TOP_TASKS_LIMIT = 20
UNKNOWN_LAB = "unknown_lab"
UNKNOWN_ROBOT = "unknown_robot"
UNKNOWN_POLICY = "unknown_policy"
UNKNOWN_CATEGORY = "unknown"
UNKNOWN_SEVERITY = "unknown"
DAY_MISSING_TIMESTAMP = "unrecorded"

CENTRAL_TZ = ZoneInfo("America/Chicago")


def now_timestamps() -> dict[str, str]:
    """Return current time as UTC and US Central (CDT/CST) ISO strings."""
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_central = now_utc.astimezone(CENTRAL_TZ)
    return {
        "utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cdt": now_central.isoformat(),
    }


_ACTION_KEYS = (
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
    "gripper_position",
    "gripper_velocity",
    "gripper_binary",
    "base_position",
    "base_velocity",
)


def _s(v):
    """Decode an HDF5 attr value to a python str (or None)."""
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _s(v.item()) if v.size else None
    if isinstance(v, (np.bytes_, np.str_)):
        return _s(v.tolist())
    return str(v)


def _attr(g, key):
    return g.attrs[key] if key in g.attrs else None


def _profile(raw):
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    return []


def _n_timesteps(group):
    if group is None:
        return None
    for k in _ACTION_KEYS:
        ds = group.get(k)
        if isinstance(ds, h5py.Dataset) and ds.shape and ds.shape[0] > 0:
            return int(ds.shape[0])
    return None


def _iso(ts):
    if ts is None:
        return None
    try:
        f = float(np.asarray(ts).item())
    except (TypeError, ValueError):
        return _s(ts)
    try:
        return (
            datetime.fromtimestamp(f, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return None


def _extract_record_from_h5(f: h5py.File, repo_path: str) -> dict:
    profile = _profile(_s(_attr(f, "robot_profile")))

    ann_group = f.get("episode_annotations")
    annotations = []
    if isinstance(ann_group, h5py.Group):
        for name, g in ann_group.items():
            if not isinstance(g, h5py.Group):
                continue
            tax_raw = _s(_attr(g, "taxonomy"))
            tax = {}
            if tax_raw:
                try:
                    decoded = json.loads(tax_raw)
                    if isinstance(decoded, dict):
                        tax = decoded
                except (json.JSONDecodeError, TypeError):
                    pass
            success = _attr(g, "success")
            desc = _s(_attr(g, "failure_description")) or ""
            annotations.append(
                {
                    "name": str(name),
                    "timestamp": _s(_attr(g, "timestamp")),
                    "success": float(np.asarray(success).item())
                    if success is not None
                    else None,
                    "failure_category": _s(tax.get("failure_category"))
                    or UNKNOWN_CATEGORY,
                    "severity": _s(tax.get("severity")) or UNKNOWN_SEVERITY,
                    "has_failure_description": bool(desc.strip()),
                }
            )

    actions = f.get("actions") if isinstance(f.get("actions"), h5py.Group) else None
    obs = f.get("observations") if isinstance(f.get("observations"), h5py.Group) else None
    video_paths = (
        obs.get("video_paths")
        if isinstance(obs, h5py.Group) and isinstance(obs.get("video_paths"), h5py.Group)
        else None
    )

    try:
        control_freq = float(profile.get("control_freq")) if profile.get("control_freq") else None
    except (TypeError, ValueError):
        control_freq = None
    n = _n_timesteps(actions)
    duration = (n / control_freq) if (n and control_freq) else None

    return {
        "repo_path": repo_path,
        "lab_id": _s(_attr(f, "lab_id")) or UNKNOWN_LAB,
        "operator_name": _s(_attr(f, "operator_name")),
        "language_instruction": _s(_attr(f, "language_instruction")),
        "timestamp": _iso(_attr(f, "timestamp")),
        "robot_name": _s(profile.get("robot_name")) or UNKNOWN_ROBOT,
        "policy_name": _s(profile.get("policy_name")) or UNKNOWN_POLICY,
        "action_space": _str_list(profile.get("action_space")),
        "cameras_present": sorted(video_paths.keys()) if video_paths else [],
        "duration_seconds": duration,
        "annotations": annotations,
    }


def extract_record(source: bytes | Path, repo_path: str) -> dict:
    """Pull the per-episode metadata we need for aggregation."""
    if isinstance(source, Path):
        with h5py.File(source, "r") as f:
            return _extract_record_from_h5(f, repo_path)
    with h5py.File(BytesIO(source), "r") as f:
        return _extract_record_from_h5(f, repo_path)


def _primary_annotation(anns: list[dict]) -> dict | None:
    if not anns:
        return None
    return max(anns, key=lambda a: (1 if a.get("timestamp") else 0, a.get("timestamp") or ""))


def aggregate(records: list[dict], repo: str) -> dict:
    per_lab_eps = Counter()
    per_lab_succ = Counter()
    per_lab_fail = Counter()
    per_lab_dur = defaultdict(float)
    per_lab_robots = defaultdict(set)
    per_lab_policies = defaultdict(set)
    per_lab_tasks = defaultdict(set)
    per_lab_first = {}
    per_lab_last = {}

    per_robot_eps = Counter()
    per_robot_dur = defaultdict(float)
    per_robot_labs = defaultdict(set)

    per_policy_eps = Counter()
    per_task_eps = Counter()
    failure_categories = Counter()
    severity = Counter()
    action_spaces = Counter()
    camera_setups = Counter()
    timeline = Counter()
    timeline_per_lab: Counter = Counter()

    per_lab_desc = Counter()
    per_dataset_eps = Counter()
    per_dataset_succ = Counter()
    per_dataset_fail = Counter()
    per_dataset_dur = defaultdict(float)
    per_dataset_desc = Counter()
    per_dataset_lab_eps: defaultdict[str, Counter] = defaultdict(Counter)
    total_dur = 0.0
    annotated = successes = failures = failure_descriptions = 0

    for r in records:
        lab = r["lab_id"]
        dataset = r.get("dataset_repo") or repo
        per_dataset_eps[dataset] += 1
        per_dataset_lab_eps[dataset][lab] += 1
        if r["duration_seconds"]:
            per_dataset_dur[dataset] += r["duration_seconds"]
        per_lab_eps[lab] += 1
        if r["duration_seconds"]:
            per_lab_dur[lab] += r["duration_seconds"]
            total_dur += r["duration_seconds"]
        per_lab_robots[lab].add(r["robot_name"])
        per_lab_policies[lab].add(r["policy_name"])
        if r["language_instruction"]:
            per_lab_tasks[lab].add(r["language_instruction"])
            per_task_eps[r["language_instruction"]] += 1

        per_robot_eps[r["robot_name"]] += 1
        per_robot_labs[r["robot_name"]].add(lab)
        if r["duration_seconds"]:
            per_robot_dur[r["robot_name"]] += r["duration_seconds"]

        per_policy_eps[r["policy_name"]] += 1

        for a in r["action_space"]:
            action_spaces[a] += 1

        if r["cameras_present"]:
            camera_setups[", ".join(r["cameras_present"])] += 1

        ts = r["timestamp"]
        if ts:
            if lab not in per_lab_first or ts < per_lab_first[lab]:
                per_lab_first[lab] = ts
            if lab not in per_lab_last or ts > per_lab_last[lab]:
                per_lab_last[lab] = ts
            day = ts[:10]
        else:
            day = DAY_MISSING_TIMESTAMP
        timeline[day] += 1
        timeline_per_lab[(day, lab)] += 1

        primary = _primary_annotation(r["annotations"])
        if primary is not None and primary["success"] is not None:
            annotated += 1
            if primary.get("has_failure_description"):
                failure_descriptions += 1
                per_lab_desc[lab] += 1
                per_dataset_desc[dataset] += 1
            if primary["success"] >= 0.5:
                successes += 1
                per_lab_succ[lab] += 1
                per_dataset_succ[dataset] += 1
            else:
                failures += 1
                per_lab_fail[lab] += 1
                per_dataset_fail[dataset] += 1
                failure_categories[primary["failure_category"]] += 1
                severity[primary["severity"]] += 1

    submission_timeline = []
    cumulative = 0
    for date, count in sorted(timeline.items()):
        cumulative += count
        submission_timeline.append({"date": date, "episodes": count, "cumulative": cumulative})

    submission_timeline_per_lab = [
        {"date": date, "lab_id": lab_id, "episodes": int(count)}
        for (date, lab_id), count in sorted(timeline_per_lab.items())
    ]

    per_lab = {
        lab: {
            "episodes": int(per_lab_eps[lab]),
            "successes": int(per_lab_succ[lab]),
            "failures": int(per_lab_fail[lab]),
            "annotated_episodes": int(per_lab_succ[lab]) + int(per_lab_fail[lab]),
            "failure_descriptions": int(per_lab_desc[lab]),
            "duration_hours": round(per_lab_dur[lab] / 3600.0, 3),
            "robots": sorted(per_lab_robots[lab]),
            "policies": sorted(per_lab_policies[lab]),
            "task_count": len(per_lab_tasks[lab]),
            "first_submission": per_lab_first.get(lab),
            "last_submission": per_lab_last.get(lab),
        }
        for lab in sorted(per_lab_eps)
    }

    per_robot = {
        robot: {
            "episodes": int(per_robot_eps[robot]),
            "labs": len(per_robot_labs[robot]),
            "duration_hours": round(per_robot_dur[robot] / 3600.0, 3),
        }
        for robot in sorted(per_robot_eps)
    }

    per_policy = {p: {"episodes": int(c)} for p, c in sorted(per_policy_eps.items())}

    per_dataset = {}
    for dataset in sorted(per_dataset_eps):
        lab_ids = {lab: int(c) for lab, c in sorted(per_dataset_lab_eps[dataset].items())}
        per_dataset[dataset] = {
            "episodes": int(per_dataset_eps[dataset]),
            "successes": int(per_dataset_succ[dataset]),
            "failures": int(per_dataset_fail[dataset]),
            "annotated_episodes": int(per_dataset_succ[dataset]) + int(per_dataset_fail[dataset]),
            "failure_descriptions": int(per_dataset_desc[dataset]),
            "duration_hours": round(per_dataset_dur[dataset] / 3600.0, 3),
            "lab_ids": lab_ids,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_timestamps(),
        "dataset_repo": repo,
        "totals": {
            "episodes": len(records),
            "labs": len(per_lab_eps),
            "datasets": len(per_dataset_eps),
            "robots": len(per_robot_eps),
            "policies": len(per_policy_eps),
            "tasks": len(per_task_eps),
            "duration_hours": round(total_dur / 3600.0, 3),
            "annotated_episodes": annotated,
            "successes": successes,
            "failures": failures,
            "failure_descriptions": failure_descriptions,
            "read_errors": 0,
        },
        "per_lab": per_lab,
        "per_dataset": per_dataset,
        "per_robot": per_robot,
        "per_policy": per_policy,
        "top_tasks": [
            {"instruction": t, "episodes": c}
            for t, c in per_task_eps.most_common(TOP_TASKS_LIMIT)
        ],
        "failure_categories": dict(failure_categories.most_common()),
        "severity_distribution": dict(severity.most_common()),
        "action_spaces": dict(action_spaces.most_common()),
        "camera_setups": dict(camera_setups.most_common()),
        "submission_timeline": submission_timeline,
        "submission_timeline_per_lab": submission_timeline_per_lab,
    }
