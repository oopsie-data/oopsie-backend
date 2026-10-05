"""Action-space data quality checks: read real per-timestep ``actions/*`` and
``observations/robot_states/*`` arrays and flag suspicious episodes.

Unlike the rest of the stats pipeline (``episode_stats.py``), which only reads HDF5
attrs for speed, these checks load full per-timestep arrays - expensive, so they run
as a separate opt-in pass (``oopsie_backend/scripts/run_action_qa.py``), not as part
of the nightly ``compute_local_stats``/``compute_hf_stats`` run.

See ``reports/data_quality_exceptions.md``'s "Action QA" / "Action vs. Observations QA"
sections for what each check means in plain language. Thresholds live in
``configs/stats_config.yaml`` under ``action_qa``.
"""

from __future__ import annotations

import json
from itertools import combinations

import h5py
import numpy as np

# Position-type action keys eligible for the scale/tracking/delta checks.
# Deliberately excludes base_position (documented delta exception,
# oopsie-docs/docs/format.md:144 - "deltas usually") and all *_velocity /
# gripper_binary keys (derivative-like by nature, not a policy violation).
POSITION_ACTION_KEYS = ("joint_position", "cartesian_position", "gripper_position")

# All nine canonical keys oopsie-data-tools' conversion_utils.py always writes
# (undeclared keys are h5py.Empty rather than omitted).
ALL_ACTION_KEYS = (
    "joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity",
    "gripper_position", "gripper_velocity", "gripper_binary",
    "base_position", "base_velocity",
)


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _decode(v.flat[0]) if v.size else ""
    return str(v)


def load_action_state_arrays(
    f: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Read every non-Empty ``actions/<key>`` and ``observations/robot_states/<key>``
    as float64 ``(T, D)``. Returns ``(actions, states, declared_action_space)`` -
    ``declared_action_space`` read from the ``robot_profile`` attr JSON.
    """
    try:
        profile = json.loads(_decode(f.attrs.get("robot_profile", "{}")))
    except (json.JSONDecodeError, TypeError):
        profile = {}
    declared = profile.get("action_space") or []
    if isinstance(declared, str):
        declared = [declared]
    declared = [str(k) for k in declared]

    actions: dict[str, np.ndarray] = {}
    actions_group = f.get("actions")
    if isinstance(actions_group, h5py.Group):
        for k in ALL_ACTION_KEYS:
            ds = actions_group.get(k)
            if isinstance(ds, h5py.Dataset) and ds.shape is not None:  # skip h5py.Empty
                actions[k] = np.asarray(ds[()], dtype=np.float64)

    states: dict[str, np.ndarray] = {}
    obs_group = f.get("observations")
    robot_states_group = obs_group.get("robot_states") if isinstance(obs_group, h5py.Group) else None
    if isinstance(robot_states_group, h5py.Group):
        for k in robot_states_group:
            ds = robot_states_group[k]
            if isinstance(ds, h5py.Dataset) and ds.shape is not None:
                states[k] = np.asarray(ds[()], dtype=np.float64)

    return actions, states, declared


def find_duplicate_action_pairs(
    actions: dict[str, np.ndarray],
    declared_keys: list[str],
    *,
    atol: float = 1e-9,
    rtol: float = 0.0,
) -> list[dict]:
    """Compare every same-shaped pair of declared action arrays for exact/near-exact
    equality. One entry per *compared* pair (mismatched shapes are never compared,
    not reported as "not flagged")."""
    results = []
    keys = sorted({k for k in declared_keys if k in actions})
    for key_a, key_b in combinations(keys, 2):
        a, b = actions[key_a], actions[key_b]
        if a.shape != b.shape:
            continue
        max_abs_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        flagged = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        results.append(
            {
                "key_a": key_a,
                "key_b": key_b,
                "shape": list(a.shape),
                "flagged": flagged,
                "max_abs_diff": max_abs_diff,
            }
        )
    return results


def _align_T(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Truncate both to ``min(T_action, T_state)``; always records whether they
    differed, regardless of any config toggle."""
    t_action, t_state = action.shape[0], state.shape[0]
    t = min(t_action, t_state)
    info = {"action_T": t_action, "state_T": t_state, "truncated": t_action != t_state}
    return action[:t], state[:t], info


def scale_consistency(action: np.ndarray, state_next: np.ndarray) -> dict:
    """``RMS(action) / RMS(state_next)`` - no thresholding here, caller applies
    config bounds."""
    action_rms = float(np.sqrt(np.mean(action**2))) if action.size else 0.0
    state_rms = float(np.sqrt(np.mean(state_next**2))) if state_next.size else 0.0
    ratio = action_rms / state_rms if state_rms > 0 else float("inf")
    return {"action_rms": action_rms, "state_rms": state_rms, "ratio": ratio}


def delta_vs_absolute_residuals(action: np.ndarray, state: np.ndarray) -> dict:
    """``resid_abs[t] = action[t] - state[t+1]`` (absolute-action hypothesis);
    ``resid_delta[t] = action[t] - (state[t+1] - state[t])`` (delta-action
    hypothesis). Computed over ``t = 0..T-2`` (``action``/``state`` must already be
    T-aligned by the caller, e.g. via :func:`_align_T`). Returns the RMS of both
    plus ``ratio = RMS(resid_delta) / RMS(resid_abs)`` - this same ``resid_abs`` is
    also what the tracking-residual check (a separate, simpler plausibility check)
    uses; computing it once here serves both.
    """
    if action.shape[0] < 2:
        return {"resid_abs_rms": 0.0, "resid_delta_rms": 0.0, "ratio": float("nan"), "action_rms": 0.0}
    a = action[:-1]
    resid_abs = a - state[1:]
    resid_delta = a - (state[1:] - state[:-1])
    resid_abs_rms = float(np.sqrt(np.mean(resid_abs**2)))
    resid_delta_rms = float(np.sqrt(np.mean(resid_delta**2)))
    action_rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
    ratio = resid_delta_rms / resid_abs_rms if resid_abs_rms > 0 else float("inf")
    return {
        "resid_abs_rms": resid_abs_rms,
        "resid_delta_rms": resid_delta_rms,
        "ratio": ratio,
        "action_rms": action_rms,
    }


def run_action_qa_on_episode(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    declared_action_space: list[str],
    config: dict,
) -> dict:
    """Orchestrate all checks per ``config[...]['enabled']`` toggles.

    Never raises - NaN/Inf/missing-key cases become a skip reason, not an
    exception, so a batch run never dies on one bad file. Returns
    ``{"findings": [...], "skips": [...]}``; empty lists when nothing applies or
    nothing was flagged.
    """
    findings: list[dict] = []
    skips: list[dict] = []

    dup_cfg = config["duplicate_pairs"]
    if dup_cfg["enabled"] and len(declared_action_space) >= 2:
        for pair in find_duplicate_action_pairs(
            actions, declared_action_space, atol=dup_cfg["atol"], rtol=dup_cfg["rtol"]
        ):
            if pair["flagged"]:
                findings.append({"test": "duplicate_pairs", **pair})

    scale_cfg = config["scale_consistency"]
    track_cfg = config["tracking_residual"]
    delta_cfg = config["delta_detection"]

    for key in declared_action_space:
        if key not in POSITION_ACTION_KEYS:
            continue
        if key not in actions:
            skips.append({"key": key, "reason": "declared_but_empty"})
            continue
        if key not in states:
            skips.append({"key": key, "reason": "no_matching_state"})
            continue

        action, state = actions[key], states[key]
        if action.shape[1:] != state.shape[1:]:
            # Different feature dimension (not just a different episode length) -
            # e.g. an action commands fewer/more DOF than the state reports for
            # some hand/gripper profiles. Elementwise comparison is meaningless
            # here; skip rather than guessing which columns correspond.
            skips.append(
                {"key": key, "reason": "feature_dim_mismatch", "action_shape": list(action.shape), "state_shape": list(state.shape)}
            )
            continue
        if not (np.all(np.isfinite(action)) and np.all(np.isfinite(state))):
            skips.append({"key": key, "reason": "non_finite_values"})
            continue

        action_t, state_t, align_info = _align_T(action, state)
        if align_info["truncated"]:
            findings.append({"test": "length_mismatch", "key": key, **align_info})
        if action_t.shape[0] < 2:
            skips.append({"key": key, "reason": "too_short"})
            continue

        if scale_cfg["enabled"]:
            sc = scale_consistency(action_t, state_t)
            if sc["ratio"] > scale_cfg["max_ratio"] or sc["ratio"] < scale_cfg["min_ratio"]:
                findings.append({"test": "scale_consistency", "key": key, **sc})

        if track_cfg["enabled"] or delta_cfg["enabled"]:
            dv = delta_vs_absolute_residuals(action_t, state_t)
            if track_cfg["enabled"] and dv["action_rms"] > 0:
                normalized_rms = dv["resid_abs_rms"] / dv["action_rms"]
                if normalized_rms > track_cfg["max_normalized_rms"]:
                    findings.append(
                        {"test": "tracking_residual", "key": key, "normalized_rms": normalized_rms, **dv}
                    )
            # action_rms == 0 (a constant/empty action channel) makes resid_delta
            # trivially small relative to resid_abs purely because there's no real
            # signal to compare - that's a "degenerate action" problem, already
            # caught distinctly by scale_consistency's min_ratio, not a delta
            # encoding. Don't double-report it here under the wrong label.
            if delta_cfg["enabled"] and dv["action_rms"] > 0 and dv["ratio"] < delta_cfg["ratio_threshold"]:
                findings.append({"test": "delta_detection", "key": key, **dv})

    return {"findings": findings, "skips": skips}


def run_action_qa_on_h5(f: h5py.File, config: dict) -> dict | None:
    """:func:`load_action_state_arrays` + :func:`run_action_qa_on_episode`.
    ``None`` if the episode has no declared action space at all (nothing to check).
    """
    actions, states, declared = load_action_state_arrays(f)
    if not declared:
        return None
    return run_action_qa_on_episode(actions, states, declared, config)
