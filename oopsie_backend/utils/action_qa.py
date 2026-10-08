"""Action-space data quality checks: read real per-timestep ``actions/*`` and
``observations/robot_states/*`` arrays and flag suspicious episodes.

Unlike the rest of the stats pipeline (``episode_stats.py``), which only reads HDF5
attrs for speed, these checks load full per-timestep arrays - expensive, so they run
as a separate opt-in pass (``oopsie_backend/scripts/run_action_qa.py``), not as part
of the nightly ``compute_local_stats``/``compute_hf_stats`` run.

Checks are grouped into categories matching ``configs/action_qa_config.yaml``'s
sections: ``general`` (any action space) plus one per arm action space
(``joint_position``, ``joint_velocity``, ``cartesian_position``, ``cartesian_velocity``).
Every finding/skip carries its ``category`` and ``test``. See
``reports/data_quality_exceptions.md`` for what each check means in plain language.
"""

from __future__ import annotations

import json
from itertools import combinations

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from oopsie_data_tools.utils.robot_profile.rotation_utils import RotOption

# Position-type keys for the general scale/tracking/delta checks. Excludes
# cartesian_position (one RMS over metres and quaternion components is meaningless - its
# own category checks position and rotation separately), base_position (documented delta
# exception, oopsie-docs/docs/format.md:144 - "deltas usually") and all *_velocity /
# gripper_binary keys (derivative-like by nature, not a policy violation).
GENERAL_POSITION_KEYS = ("joint_position", "gripper_position")

# All nine canonical keys oopsie-data-tools' conversion_utils.py always writes
# (undeclared keys are h5py.Empty rather than omitted).
ALL_ACTION_KEYS = (
    "joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity",
    "gripper_position", "gripper_velocity", "gripper_binary",
    "base_position", "base_velocity",
)

# The state each action key is observed through (velocity commands via the position state).
STATE_KEY_FOR_ACTION = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_position",
    "cartesian_position": "cartesian_position",
    "cartesian_velocity": "cartesian_position",
    "gripper_position": "gripper_position",
    "gripper_velocity": "gripper_position",
    "gripper_binary": "gripper_position",
    "base_position": "base_position",
    "base_velocity": "base_position",
}

# Width of each orientation representation in a flat pose vector.
_ROT_WIDTH = {RotOption.QUAT: 4, RotOption.MATRIX: 9, RotOption.ROT6D: 6, RotOption.ROTVEC: 3}


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _decode(v.flat[0]) if v.size else ""
    return str(v)


def load_action_state_arrays(
    f: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Read every non-Empty ``actions/<key>`` and ``observations/robot_states/<key>``
    as float64 ``(T, D)``. Returns ``(actions, states, robot_profile)``, the profile
    parsed from the ``robot_profile`` attr JSON (``{}`` if missing or unparseable).
    """
    try:
        profile = json.loads(_decode(f.attrs.get("robot_profile", "{}")))
    except (json.JSONDecodeError, TypeError):
        profile = {}
    if not isinstance(profile, dict):
        profile = {}

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

    return actions, states, profile


def _finding(category: str, test: str, key: str, **details) -> dict:
    return {"category": category, "test": test, "key": key, **details}


def _skip(category: str, test: str, key: str, reason: str, **details) -> dict:
    return {"category": category, "test": test, "key": key, "reason": reason, **details}


def _params(check_cfg: dict) -> dict:
    """A check's config minus its ``enabled`` toggle, for passing as keyword arguments."""
    return {k: v for k, v in check_cfg.items() if k != "enabled"}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


# ── general ───────────────────────────────────────────────────────────────────


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


def _align_T(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both to ``min(T_action, T_state)`` (a mismatch is reported separately by
    the general ``length_mismatch`` check)."""
    t = min(action.shape[0], state.shape[0])
    return action[:t], state[:t]


def scale_consistency(action: np.ndarray, state_next: np.ndarray) -> dict:
    """``RMS(action) / RMS(state_next)`` - no thresholding here, caller applies
    config bounds."""
    action_rms = _rms(action)
    state_rms = _rms(state_next)
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
    resid_abs_rms = _rms(a - state[1:])
    resid_delta_rms = _rms(a - (state[1:] - state[:-1]))
    ratio = resid_delta_rms / resid_abs_rms if resid_abs_rms > 0 else float("inf")
    return {
        "resid_abs_rms": resid_abs_rms,
        "resid_delta_rms": resid_delta_rms,
        "ratio": ratio,
        "action_rms": _rms(a),
    }


def _run_general_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    declared: list[str],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
) -> None:
    category = "general"

    if cfg["duplicate_pairs"]["enabled"] and len(declared) >= 2:
        dup_cfg = cfg["duplicate_pairs"]
        checked.add((category, "duplicate_pairs"))
        for pair in find_duplicate_action_pairs(actions, declared, atol=dup_cfg["atol"], rtol=dup_cfg["rtol"]):
            if pair["flagged"]:
                findings.append(_finding(category, "duplicate_pairs", f"{pair['key_a']}+{pair['key_b']}", **pair))

    if cfg["length_mismatch"]["enabled"]:
        for key in declared:
            state_key = STATE_KEY_FOR_ACTION.get(key)
            if key not in actions or state_key not in states:
                continue
            checked.add((category, "length_mismatch"))
            action_T, state_T = actions[key].shape[0], states[state_key].shape[0]
            if action_T != state_T:
                findings.append(
                    _finding(category, "length_mismatch", key, state_key=state_key, action_T=action_T, state_T=state_T)
                )

    tests = [t for t in ("scale_consistency", "tracking_residual", "delta_detection") if cfg[t]["enabled"]]
    if not tests:
        return

    for key in declared:
        if key not in GENERAL_POSITION_KEYS:
            continue

        def skip(reason: str, **details) -> None:
            skips.extend(_skip(category, t, key, reason, **details) for t in tests)

        if key not in actions:
            skip("declared_but_empty")
            continue
        if key not in states:
            skip("no_matching_state")
            continue

        action, state = actions[key], states[key]
        if action.shape[1:] != state.shape[1:]:
            # Different feature dimension (not just a different episode length) -
            # e.g. an action commands fewer/more DOF than the state reports for
            # some hand/gripper profiles. Elementwise comparison is meaningless
            # here; skip rather than guessing which columns correspond.
            skip("feature_dim_mismatch", action_shape=list(action.shape), state_shape=list(state.shape))
            continue
        if not (np.all(np.isfinite(action)) and np.all(np.isfinite(state))):
            skip("non_finite_values")
            continue

        action_t, state_t = _align_T(action, state)
        if action_t.shape[0] < 2:
            skip("too_short")
            continue

        checked.update((category, t) for t in tests)
        if "scale_consistency" in tests:
            scale_cfg = cfg["scale_consistency"]
            sc = scale_consistency(action_t, state_t)
            if sc["ratio"] > scale_cfg["max_ratio"] or sc["ratio"] < scale_cfg["min_ratio"]:
                findings.append(_finding(category, "scale_consistency", key, **sc))

        if "tracking_residual" in tests or "delta_detection" in tests:
            dv = delta_vs_absolute_residuals(action_t, state_t)
            if "tracking_residual" in tests and dv["action_rms"] > 0:
                normalized_rms = dv["resid_abs_rms"] / dv["action_rms"]
                if normalized_rms > cfg["tracking_residual"]["max_normalized_rms"]:
                    findings.append(_finding(category, "tracking_residual", key, normalized_rms=normalized_rms, **dv))
            # action_rms == 0 (a constant/empty action channel) makes resid_delta
            # trivially small relative to resid_abs purely because there's no real
            # signal to compare - that's a "degenerate action" problem, already
            # caught distinctly by scale_consistency's min_ratio, not a delta
            # encoding. Don't double-report it here under the wrong label.
            if (
                "delta_detection" in tests
                and dv["action_rms"] > 0
                and dv["ratio"] < cfg["delta_detection"]["ratio_threshold"]
            ):
                findings.append(_finding(category, "delta_detection", key, **dv))


# ── cartesian_position ────────────────────────────────────────────────────────


def _looks_scalar_first(quat: np.ndarray) -> bool:
    """Heuristic only - unit norm is invariant to component order, so this can't be a
    hard guarantee. Across most real trajectories (small/moderate rotations away from
    identity), the scalar (w) component stays closer to +-1 with lower variance than the
    vector (x, y, z) components, which center near 0. Compares how "scalar-like" index 0
    looks vs index 3 of the 4-slot quaternion; True suggests ``[qw, qx, qy, qz]`` was
    written where scalar-last ``[qx, qy, qz, qw]`` was required.
    """
    abs_mean = np.mean(np.abs(quat), axis=0)
    std = np.std(quat, axis=0)
    scalar_likeness = abs_mean - std  # higher = closer to a near-constant +-1 value
    return bool(scalar_likeness[0] > scalar_likeness[3])


def validate_cartesian_quaternion_shape(
    action: np.ndarray, *, is_biarm: bool, norm_atol: float = 1e-3, check_scalar_order: bool = True
) -> dict:
    """Check a ``cartesian_position`` action array against the mandated per-arm
    ``[x, y, z, qx, qy, qz, qw]`` (scalar-last) layout: dim must be 7 (single-arm) or 14
    (biarm), each arm's 4-slot quaternion must be unit-norm, and (heuristically, see
    :func:`_looks_scalar_first`) the scalar component should be last, not first. Mirrors
    the upload-time validator's ``validate_cartesian_quaternions`` (oopsie-data-tools)
    for dim/norm, re-checked here independently since that validator only ever ran on
    files that went through it; the scalar-order check has no upstream equivalent.

    Returns ``{"ok": True}`` or ``{"ok": False, "reason": ..., **details}``.
    """
    expected_dim = 14 if is_biarm else 7
    dim = action.shape[-1] if action.ndim else 0
    if dim != expected_dim:
        return {"ok": False, "reason": "wrong_dim", "expected_dim": expected_dim, "actual_dim": dim}

    starts = (3, 10) if is_biarm else (3,)
    bad_arms = []
    scalar_first_arms = []
    for arm_index, start in enumerate(starts, 1):
        quat = action[..., start : start + 4]
        norms = np.linalg.norm(quat, axis=-1)
        valid = np.isclose(norms, 1.0, rtol=0.0, atol=norm_atol)
        if not np.all(valid):
            bad_norms = norms[~valid]
            bad_arms.append(
                {"arm": arm_index, "min_norm": float(np.min(bad_norms)), "max_norm": float(np.max(bad_norms))}
            )
            continue  # not a unit quaternion at all - scalar-order heuristic is meaningless here
        if check_scalar_order and quat.shape[0] >= 2 and _looks_scalar_first(quat):
            scalar_first_arms.append(arm_index)

    if bad_arms:
        return {"ok": False, "reason": "bad_quaternion_norm", "bad_arms": bad_arms}
    if scalar_first_arms:
        return {"ok": False, "reason": "suspected_scalar_first", "arms": scalar_first_arms}
    return {"ok": True}


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row rotation angle between unit quaternions ``a`` and ``b`` (degrees),
    sign-invariant (q and -q are the same rotation)."""
    dots = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _rot_width(option: RotOption) -> int:
    return _ROT_WIDTH.get(option, 3)  # every Euler order is 3 wide


def _state_rotation_to_quat(rot: np.ndarray, declared: str | None) -> np.ndarray | None:
    """Convert one arm's ``(T, k)`` state rotation slice to scalar-last quaternions.

    The recorder converts states with a declared ``robot_state_orientation_representation``
    to quaternions on write, so a 4-wide slice is taken as a quaternion whatever the profile
    says; any other width is converted using the declared representation if its width
    matches. ``None`` when the representation can't be determined or converted.
    """
    width = rot.shape[1]
    option = None
    if declared:
        try:
            parsed = RotOption.from_string(declared)
        except ValueError:
            parsed = None
        if parsed is not None and _rot_width(parsed) == width:
            option = parsed
    if option is None and width == 4:
        option = RotOption.QUAT
    if option is None:
        return None

    try:
        if option == RotOption.QUAT:
            norms = np.linalg.norm(rot, axis=1, keepdims=True)
            return rot / norms if np.all(norms > 0) else None
        if option == RotOption.MATRIX:
            return Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
        if option == RotOption.ROT6D:
            # First two rotation-matrix columns; rebuild the third by Gram-Schmidt
            # (same convention as oopsie-data-tools' ActionQuatConversion).
            cols = rot.reshape(-1, 2, 3)
            u1 = cols[:, 0] / np.linalg.norm(cols[:, 0], axis=1, keepdims=True)
            u2 = cols[:, 1] - np.sum(cols[:, 1] * u1, axis=1, keepdims=True) * u1
            u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
            return Rotation.from_matrix(np.stack([u1, u2, np.cross(u1, u2)], axis=-1)).as_quat()
        if option == RotOption.ROTVEC:
            return Rotation.from_rotvec(rot).as_quat()
        return Rotation.from_euler(option.name, rot).as_quat()
    except ValueError:
        return None


# Smaller units a metre position is commonly mistaken for, and their size in metres.
_UNIT_FACTORS = {"cm": 100.0, "mm": 1000.0}


def _unit_from_ratio(ratio: float) -> str | None:
    """Unit whose metre factor is within half a decade of the action/state scale ratio
    (so ~32-316x reads as cm, ~316-3162x as mm), assuming the state is in metres."""
    for unit, factor in _UNIT_FACTORS.items():
        if abs(np.log10(ratio / factor)) < 0.5:
            return unit
    return None


def _unit_from_reach(max_abs: float, reach_min_m: float, reach_max_m: float) -> str | None:
    """Unit that, converted to metres, puts ``max_abs`` within a typical arm reach;
    when both do, the one landing nearest the middle of that range (log scale)."""
    center = np.sqrt(reach_min_m * reach_max_m)
    fits = {
        unit: abs(np.log(max_abs / factor / center))
        for unit, factor in _UNIT_FACTORS.items()
        if reach_min_m <= max_abs / factor <= reach_max_m
    }
    return min(fits, key=fits.get) if fits else None


def check_position_units(
    action_xyz: np.ndarray,
    state_xyz: np.ndarray | None,
    *,
    max_abs_position_m: float,
    max_action_state_ratio: float,
    typical_reach_min_m: float,
    typical_reach_max_m: float,
) -> dict:
    """Positions must be in metres: flags action (and state) xyz beyond
    ``max_abs_position_m`` - mm/cm values or a far-away world frame - and an action xyz
    RMS more than ``max_action_state_ratio`` times the state's (action in smaller units
    than the state). The low side is left to delta detection, since delta-encoded
    actions are also much smaller than the state.

    Each flagged side gets a ``*_suspected_unit`` guess (``"cm"``, ``"mm"`` or ``None``):
    from the action/state ratio when the state looks like metres, otherwise from which
    unit puts the values within a typical arm reach.

    ``*_xyz`` hold every arm's xyz columns. Returns ``{"ok": bool, "reasons": [...], ...}``.
    """
    reasons = []
    result: dict = {"action_max_abs_m": float(np.max(np.abs(action_xyz)))}
    if result["action_max_abs_m"] > max_abs_position_m:
        reasons.append("action_out_of_range")
    state_in_metres = False
    if state_xyz is not None:
        result["state_max_abs_m"] = float(np.max(np.abs(state_xyz)))
        if result["state_max_abs_m"] > max_abs_position_m:
            reasons.append("state_out_of_range")
        else:
            state_in_metres = True
        state_rms = _rms(state_xyz)
        result["action_state_ratio"] = _rms(action_xyz) / state_rms if state_rms > 0 else float("inf")
        if result["action_state_ratio"] > max_action_state_ratio:
            reasons.append("action_state_scale_mismatch")

    if "action_out_of_range" in reasons or "action_state_scale_mismatch" in reasons:
        if state_in_metres and np.isfinite(result["action_state_ratio"]):
            result["action_suspected_unit"] = _unit_from_ratio(result["action_state_ratio"])
        else:
            result["action_suspected_unit"] = _unit_from_reach(
                result["action_max_abs_m"], typical_reach_min_m, typical_reach_max_m
            )
    if "state_out_of_range" in reasons:
        result["state_suspected_unit"] = _unit_from_reach(
            result["state_max_abs_m"], typical_reach_min_m, typical_reach_max_m
        )
    return {"ok": not reasons, "reasons": reasons, **result}


def check_quaternion_sign_flips(arm_quats: list[np.ndarray]) -> list[dict]:
    """Per arm, steps where ``dot(q[t], q[t+1]) < 0``: the same orientation written
    with the opposite sign, which breaks learning on raw quaternion values."""
    flagged = []
    for arm, q in enumerate(arm_quats, 1):
        flips = np.flatnonzero(np.sum(q[1:] * q[:-1], axis=1) < 0)
        if flips.size:
            flagged.append({"arm": arm, "count": int(flips.size), "first_step": int(flips[0])})
    return flagged


def check_rotation_jumps(arm_quats: list[np.ndarray], max_step_deg: float) -> list[dict]:
    """Per arm, steps whose rotation to the next step exceeds ``max_step_deg``."""
    flagged = []
    for arm, q in enumerate(arm_quats, 1):
        angles = _quat_angle_deg(q[1:], q[:-1])
        jumps = np.flatnonzero(angles > max_step_deg)
        if jumps.size:
            flagged.append(
                {
                    "arm": arm,
                    "count": int(jumps.size),
                    "first_step": int(jumps[0]),
                    "max_step_deg": float(np.max(angles)),
                }
            )
    return flagged


def rotation_tracking_errors(action_quats: list[np.ndarray], state_quats: list[np.ndarray]) -> list[dict]:
    """Per arm, angle between ``action[t]`` and ``state[t+1]`` (T-aligned inputs)."""
    errors = []
    for arm, (a, s) in enumerate(zip(action_quats, state_quats), 1):
        angles = _quat_angle_deg(a[:-1], s[1:])
        errors.append(
            {
                "arm": arm,
                "median_error_deg": float(np.median(angles)),
                "mean_error_deg": float(np.mean(angles)),
                "max_error_deg": float(np.max(angles)),
            }
        )
    return errors


def biarm_swap_ratio(action_xyz: list[np.ndarray], state_xyz: list[np.ndarray]) -> dict:
    """Position error of action[t] vs state[t+1] with the arms as given and with them
    swapped (T-aligned inputs). ``ratio = swapped / same``: well below 1 means the
    action's arm order is reversed relative to the state's."""
    a1, a2 = action_xyz[0][:-1], action_xyz[1][:-1]
    s1, s2 = state_xyz[0][1:], state_xyz[1][1:]
    same = _rms(np.concatenate([a1 - s1, a2 - s2]))
    swapped = _rms(np.concatenate([a1 - s2, a2 - s1]))
    ratio = swapped / same if same > 0 else float("inf")
    return {"same_arm_rms_m": same, "swapped_arm_rms_m": swapped, "ratio": ratio}


def _run_cartesian_position_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
    *,
    is_biarm: bool,
    control_freq: float | None,
    state_orientation: str | None,
) -> None:
    category = key = "cartesian_position"
    tests = [t for t, c in cfg.items() if c["enabled"]]

    def skip(reason: str, only: list[str] | None = None, **details) -> None:
        skips.extend(_skip(category, t, key, reason, **details) for t in (tests if only is None else only))

    if not tests:
        return
    if key not in actions:
        skip("declared_but_empty")
        return
    action = actions[key]
    if not np.all(np.isfinite(action)):
        skip("non_finite_values")
        return

    if "quaternion_shape" in tests:
        checked.add((category, "quaternion_shape"))
        qcfg = cfg["quaternion_shape"]
        shape_check = validate_cartesian_quaternion_shape(
            action, is_biarm=is_biarm, norm_atol=qcfg["norm_atol"], check_scalar_order=qcfg["check_scalar_order"]
        )
        if not shape_check["ok"]:
            findings.append(_finding(category, "quaternion_shape", key, **shape_check))
        tests.remove("quaternion_shape")
    if not tests:
        return

    # Everything below relies on the mandated per-arm [x, y, z, qx, qy, qz, qw] layout;
    # a wrong width is already reported by quaternion_shape.
    n_arms = 2 if is_biarm else 1
    if action.ndim != 2 or action.shape[1] != 7 * n_arms:
        skip("wrong_action_dim", actual_shape=list(action.shape))
        return
    if action.shape[0] < 2:
        skip("too_short")
        return
    action_xyz = [action[:, 7 * a : 7 * a + 3] for a in range(n_arms)]
    action_quats = [action[:, 7 * a + 3 : 7 * a + 7] for a in range(n_arms)]
    rotation_tests = [t for t in ("quaternion_sign_flip", "rotation_jump", "rotation_tracking") if t in tests]
    if any(np.any(np.linalg.norm(q, axis=1) == 0) for q in action_quats):
        skip("zero_quaternion", rotation_tests)
        tests = [t for t in tests if t not in rotation_tests]
    else:
        action_quats = [q / np.linalg.norm(q, axis=1, keepdims=True) for q in action_quats]

    # Action-only checks.
    if "quaternion_sign_flip" in tests:
        checked.add((category, "quaternion_sign_flip"))
        flips = check_quaternion_sign_flips(action_quats)
        if flips:
            findings.append(_finding(category, "quaternion_sign_flip", key, arms=flips))

    if "rotation_jump" in tests:
        if not control_freq or control_freq <= 0:
            skip("no_control_freq", ["rotation_jump"])
        else:
            checked.add((category, "rotation_jump"))
            max_step_deg = cfg["rotation_jump"]["max_angular_speed_deg_s"] / control_freq
            jumps = check_rotation_jumps(action_quats, max_step_deg)
            if jumps:
                findings.append(
                    _finding(category, "rotation_jump", key, limit_step_deg=max_step_deg, control_freq=control_freq, arms=jumps)
                )

    # Checks against the state; position_units still runs its action-only part without one.
    state_tests = [t for t in ("rotation_tracking", "delta_detection", "biarm_ordering") if t in tests]
    if not is_biarm and "biarm_ordering" in state_tests:
        state_tests.remove("biarm_ordering")  # not applicable, not a skip

    state = states.get(key)
    state_problem = None
    if state is None:
        state_problem = {"reason": "no_matching_state"}
    elif state.ndim != 2 or state.shape[1] % n_arms or state.shape[1] // n_arms < 3:
        state_problem = {"reason": "unexpected_state_dim", "state_shape": list(state.shape)}
    elif not np.all(np.isfinite(state)):
        state_problem = {"reason": "non_finite_state"}
    if state_problem is not None:
        skip(only=state_tests, **state_problem)
        if "position_units" in tests:
            checked.add((category, "position_units"))
            units = check_position_units(
                np.concatenate(action_xyz, axis=1), None,
                **_params(cfg["position_units"]),
            )
            if not units["ok"]:
                findings.append(_finding(category, "position_units", key, **units))
        return

    arm_width = state.shape[1] // n_arms
    t = min(action.shape[0], state.shape[0])
    action_xyz = [x[:t] for x in action_xyz]
    action_quats = [q[:t] for q in action_quats]
    state = state[:t]
    state_xyz = [state[:, arm_width * a : arm_width * a + 3] for a in range(n_arms)]

    if "position_units" in tests:
        checked.add((category, "position_units"))
        units = check_position_units(
            np.concatenate(action_xyz, axis=1), np.concatenate(state_xyz, axis=1),
            **_params(cfg["position_units"]),
        )
        if not units["ok"]:
            findings.append(_finding(category, "position_units", key, **units))

    if t < 2:
        skip("too_short", state_tests)
        return

    if "delta_detection" in state_tests:
        checked.add((category, "delta_detection"))
        dv = delta_vs_absolute_residuals(np.concatenate(action_xyz, axis=1), np.concatenate(state_xyz, axis=1))
        if dv["action_rms"] > 0 and dv["ratio"] < cfg["delta_detection"]["ratio_threshold"]:
            findings.append(_finding(category, "delta_detection", key, **dv))

    if "biarm_ordering" in state_tests:
        checked.add((category, "biarm_ordering"))
        swap = biarm_swap_ratio(action_xyz, state_xyz)
        if swap["ratio"] < cfg["biarm_ordering"]["max_swap_ratio"]:
            findings.append(_finding(category, "biarm_ordering", key, **swap))

    if "rotation_tracking" in state_tests:
        state_quats = [
            _state_rotation_to_quat(state[:, arm_width * a + 3 : arm_width * (a + 1)], state_orientation)
            for a in range(n_arms)
        ]
        if any(q is None for q in state_quats):
            skip(
                "unknown_state_orientation",
                ["rotation_tracking"],
                state_rotation_width=arm_width - 3,
                declared=state_orientation,
            )
        else:
            checked.add((category, "rotation_tracking"))
            errors = rotation_tracking_errors(action_quats, state_quats)
            limit = cfg["rotation_tracking"]["max_median_error_deg"]
            if any(e["median_error_deg"] > limit for e in errors):
                findings.append(_finding(category, "rotation_tracking", key, limit_deg=limit, arms=errors))


# ── episode entry points ──────────────────────────────────────────────────────


def run_action_qa_on_episode(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    declared_action_space: list[str],
    config: dict,
    *,
    is_biarm: bool = False,
    control_freq: float | None = None,
    state_orientation: str | None = None,
) -> dict:
    """Run every enabled check, category by category, per ``config``.

    Never raises - NaN/Inf/missing-key cases become a skip reason, not an
    exception, so a batch run never dies on one bad file. Returns
    ``{"findings": [...], "skips": [...], "checked": [(category, test), ...]}`` - ``checked``
    lists the checks that actually ran, so "passed" can be told apart from "not applicable".
    """
    findings: list[dict] = []
    skips: list[dict] = []
    checked: set[tuple[str, str]] = set()

    _run_general_checks(actions, states, declared_action_space, config["general"], findings, skips, checked)
    if "cartesian_position" in declared_action_space:
        _run_cartesian_position_checks(
            actions,
            states,
            config["cartesian_position"],
            findings,
            skips,
            checked,
            is_biarm=is_biarm,
            control_freq=control_freq,
            state_orientation=state_orientation,
        )
    # joint_position / joint_velocity / cartesian_velocity: no category-specific checks yet.

    return {"findings": findings, "skips": skips, "checked": sorted(checked)}


def run_action_qa_on_h5(f: h5py.File, config: dict) -> dict | None:
    """:func:`load_action_state_arrays` + :func:`run_action_qa_on_episode`.
    ``None`` if the episode has no declared action space at all (nothing to check).
    """
    actions, states, profile = load_action_state_arrays(f)
    declared = profile.get("action_space") or []
    if isinstance(declared, str):
        declared = [declared]
    declared = [str(k) for k in declared]
    if not declared:
        return None

    try:
        control_freq = float(profile.get("control_freq"))
    except (TypeError, ValueError):
        control_freq = None
    return run_action_qa_on_episode(
        actions,
        states,
        declared,
        config,
        is_biarm=bool(profile.get("is_biarm", False)),
        control_freq=control_freq,
        state_orientation=profile.get("robot_state_orientation_representation"),
    )
