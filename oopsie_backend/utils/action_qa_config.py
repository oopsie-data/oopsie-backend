"""Load configs/action_qa_config.yaml: enable/disable toggles and thresholds for the
action-space QA checks run by run_action_qa.py. See that file for what each field means.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _BACKEND_ROOT / "configs" / "action_qa_config.yaml"

# Every check, grouped by category (the YAML's top-level check sections), with its default
# parameters. The YAML overrides any value; each value is coerced to its default's type.
DEFAULT_CHECKS: dict[str, dict[str, dict]] = {
    "general": {
        "duplicate_pairs": {"enabled": False, "atol": 1.0e-9, "rtol": 0.0},
        "length_mismatch": {"enabled": True},
        "action_copies_state": {"enabled": True, "atol": 1.0e-6, "min_fraction": 0.9},
        "scale_consistency": {"enabled": False, "max_ratio": 5.0, "min_ratio": 0.2},
        "tracking_residual": {"enabled": False, "max_normalized_rms": 0.5},
        "delta_detection": {"enabled": False, "ratio_threshold": 0.3},
    },
    "joint_position": {
        "joint_names": {"enabled": True},
        "dof": {"enabled": True},
        "radians": {"enabled": True, "max_abs_rad": 6.5},
        "step_jump": {"enabled": True, "max_joint_speed_rad_s": 6.28},
        "gripper_in_joints": {"enabled": True},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "tracking": {"enabled": True, "max_median_error_rad": 0.2},
    },
    "joint_velocity": {
        "dof": {"enabled": True},
        "speed_limit": {"enabled": True, "max_joint_speed_rad_s": 3.14},
        "normalized_range": {"enabled": True},
        "integration": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_motion_rad": 0.2,
        },
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "smoothness": {"enabled": True, "max_sign_flip_rate": 0.3},
    },
    "cartesian_position": {
        "biarm_ordering": {"enabled": True, "max_swap_ratio": 0.5},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "pose_tracking": {
            "enabled": True,
            "max_median_position_error_m": 0.10,
            "max_median_orientation_error_deg": 20.0,
        },
        "position_units": {
            "enabled": True,
            "max_abs_position_m": 3.0,
            "max_action_state_ratio": 10.0,
            "typical_reach_min_m": 0.2,
            "typical_reach_max_m": 1.5,
        },
        "quaternion_shape": {"enabled": True, "norm_atol": 1.0e-3},
        "quaternion_sign_flip": {"enabled": True},
        "rotation_jump": {"enabled": True, "max_angular_speed_deg_s": 360.0},
        "quaternion_scalar_order": {"enabled": False},
    },
    "cartesian_velocity": {
        "shape": {"enabled": True},
        "speed_limit": {"enabled": True, "max_linear_m_s": 2.0, "max_angular_rad_s": 6.28},
        "normalized_range": {"enabled": True},
        "integration": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_translation_m": 0.05,
            "min_rotation_deg": 15.0,
        },
        "linear_angular_order": {"enabled": True, "min_improvement": 0.2},
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "smoothness": {"enabled": True, "max_sign_flip_rate": 0.3},
    },
    "video": {
        "resolution": {"enabled": True, "min_short_side": 360},
        "decoding": {"enabled": True, "max_frame_count_diff": 1},
        "brightness": {
            "enabled": True,
            "dark_median": 30.0,
            "bright_median": 225.0,
            "dark_frame": 15.0,
            "bright_frame": 240.0,
            "extreme_frame_fraction": 0.1,
        },
        "speed": {
            "enabled": True,
            "fps_rel_tol": 0.05,
            "duration_tol_s": 0.5,
            "duration_rel_tol": 0.05,
            "frame_interval_cv": 0.1,
        },
        "frames": {
            "enabled": True,
            "moving_fraction_of_p95": 0.1,
            "static_diff": 0.05,
            "duplicate_diff": 0.01,
            "frozen_s": 1.0,
            "choppy_fraction": 0.2,
            "jump_diff": 40.0,
            "jump_ratio": 20.0,
            "glitch_diff": 20.0,
            "glitch_ratio": 4.0,
        },
    },
}

_NON_CHECK_SECTIONS = ("sampling", "scope", "run")


def _resolve_path(value: str | None) -> Path | None:
    """A relative path in the YAML is relative to the backend root (where this file's
    paths like REPORTS_DIR live), not the shell's cwd when the script happens to run."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (_BACKEND_ROOT / path)


def _reject_unknown(where: str, given, known) -> None:
    # A typo'd name would otherwise silently fall back to the default (e.g. leave a check on).
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"action_qa_config.yaml: unknown {where}: {unknown}")


def _build_checks(raw: dict) -> dict:
    checks: dict[str, dict] = {}
    for category, defaults in DEFAULT_CHECKS.items():
        raw_category = raw.get(category) or {}
        _reject_unknown(f"check(s) under {category!r}", raw_category, defaults)
        checks[category] = {}
        for check, params in defaults.items():
            raw_check = raw_category.get(check) or {}
            _reject_unknown(f"parameter(s) under {category}.{check}", raw_check, params)
            checks[category][check] = {
                name: type(default)(raw_check.get(name, default)) for name, default in params.items()
            }
    return checks


def _build_config(raw: dict) -> dict:
    _reject_unknown("top-level section(s)", raw, [*DEFAULT_CHECKS, *_NON_CHECK_SECTIONS])
    sampling_raw = raw.get("sampling") or {}
    scope_raw = raw.get("scope") or {}
    run_raw = raw.get("run") or {}

    return {
        **_build_checks(raw),
        "sampling": {
            "default_episode_limit": int(sampling_raw.get("default_episode_limit", 200)),
            "random": bool(sampling_raw.get("random", True)),
            "random_seed": (
                int(sampling_raw["random_seed"]) if sampling_raw.get("random_seed") is not None else None
            ),
        },
        "scope": {
            "include_stats_excluded_datasets": bool(
                scope_raw.get("include_stats_excluded_datasets", False)
            ),
            "include_public_excluded_datasets": bool(
                scope_raw.get("include_public_excluded_datasets", False)
            ),
        },
        "run": {
            "data_dir": _resolve_path(run_raw.get("data_dir")),
            "max_total_episodes": (
                int(run_raw["max_total_episodes"])
                if run_raw.get("max_total_episodes") is not None
                else None
            ),
            "output": _resolve_path(run_raw.get("output")) or (_BACKEND_ROOT / "reports" / "action_qa.json"),
            "only_dataset": [str(d) for d in (run_raw.get("only_dataset") or [])],
            "verbose": bool(run_raw.get("verbose", False)),
            "workers": int(run_raw.get("workers", 8)),
        },
    }


def load_action_qa_config(path: Path | None = None) -> dict:
    """Load and validate configs/action_qa_config.yaml into a plain dict.

    Falls back to the built-in defaults (matching the shipped YAML) if the file is
    missing, so importing this module never hard-fails.
    """
    path = path or DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return _build_config(raw)


# Loaded once at import time.
CONFIG: dict = load_action_qa_config()
