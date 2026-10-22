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
        "action_chunk_shape": {"enabled": True},
        "constant_action": {"enabled": True, "atol": 1.0e-9},
        "command_rate": {"enabled": True, "atol": 1.0e-9, "min_holds": 10, "min_fraction": 0.8},
        "required_state_keys": {"enabled": True},
        "placeholder_metadata": {
            "enabled": True,
            "placeholder_values": [
                "unknown", "none", "null", "n/a", "na", "tbd", "todo", "test", "default",
                "placeholder", "example", "xxx", "-", "?", "robot", "gripper", "policy",
            ],
            "template_prefixes": ["your_", "example_", "my_"],
            "operator_placeholder_values": [
                "operator", "pending_operator", "test_operator", "user", "admin", "human", "teleoperator",
            ],
            "operator_non_person_words": [
                "policy", "eval", "autoeval", "orchestrator", "script", "bot", "lab", "pipeline", "system", "oopsie",
            ],
        },
        "action_copies_state": {"enabled": True, "atol": 1.0e-6, "min_fraction": 0.9},
        "scale_consistency": {"enabled": False, "max_ratio": 5.0, "min_ratio": 0.2},
        "tracking_residual": {"enabled": False, "max_normalized_rms": 0.5},
        "delta_detection": {"enabled": False, "ratio_threshold": 0.3},
        "gripper_binary_present": {"enabled": True},
        "gripper_range": {"enabled": True, "atol": 0.05},
        "gripper_command_range": {"enabled": True, "max_fraction": 0.25, "min_margin": 0.05},
    },
    "joint_position": {
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "tracking": {"enabled": True, "max_median_error_rad": 0.2},
        "joint_names": {"enabled": True},
        "dof": {"enabled": True},
        "radians": {"enabled": True, "max_abs_rad": 6.5},
        "step_jump": {"enabled": True, "max_joint_speed_rad_s": 20.0},
        "gripper_in_joints": {"enabled": True},
        "normalized_range": {"enabled": True},
    },
    "joint_velocity": {
        "dof": {"enabled": True},
        "tracking": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_motion_rad": 0.2,
        },
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "normalized_range": {"enabled": True},
        "speed_limit": {"enabled": True, "max_joint_speed_rad_s": 3.14},
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
        "normalized_range": {"enabled": True},
        "quaternion_scalar_order": {"enabled": False},
    },
    "cartesian_velocity": {
        "shape": {"enabled": True},
        "speed_limit": {"enabled": True, "max_linear_m_s": 2.0, "max_angular_rad_s": 6.28},
        "normalized_range": {"enabled": True},
        "tracking": {
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
        # Ordered as the report shows them: the critical checks first.
        "decoding": {"enabled": True, "max_frame_count_diff": 1},
        "corrupted_frames": {"enabled": True, "glitch_diff": 20.0, "glitch_ratio": 4.0},
        "state_jump": {"enabled": True, "jump_ratio": 10.0, "min_fraction_of_p95": 1.0},
        "video_jump": {"enabled": True, "jump_diff": 40.0, "jump_ratio": 20.0, "state_jump_ratio": 5.0},
        "recording_gap": {"enabled": True, "video_jump_ratio": 5.0},
        "periodic_skips": {
            "enabled": True,
            "jump_ratio": 3.0,
            "min_jumps": 6,
            "min_period_score": 0.4,
            "min_coverage": 0.6,
            "min_joint_jump_rad": 0.015,
            "min_ee_jump_cm": 1.0,
            "min_video_jump": 2.0,
            "min_confirmed_fraction": 0.4,
            "command_explains_ratio": 3.0,
        },
        "choppy": {"enabled": True, "duplicate_diff": 0.01, "choppy_fraction": 0.2},
        "frozen": {"enabled": True, "static_diff": 0.05, "frozen_s": 1.0},
        "resolution": {"enabled": True, "min_short_side": 360},
        "brightness": {
            "enabled": True,
            "dark_median": 30.0,
            "bright_median": 225.0,
            "dark_frame": 15.0,
            "bright_frame": 240.0,
            "extreme_frame_fraction": 0.1,
        },
        "fps": {"enabled": True, "fps_rel_tol": 0.05},
        "duration": {"enabled": True, "duration_tol_s": 0.5, "duration_rel_tol": 0.05},
        "camera_duration_mismatch": {"enabled": True, "duration_tol_s": 0.5},
    },
}

_NON_CHECK_SECTIONS = ("sampling", "scope", "run", "report")

# How each check's findings appear in per-lab feedback, as (confidence, action):
#   confidence - "high_confidence": a flag is a real problem; "lower_confidence": a heuristic that can flag fine episodes or miss real issues.
#   action - "fix_and_resubmit": the lab fixes the data and resubmits; "confirm_and_discuss": the lab
#   confirms the issue and talks to us (unclear findings, or real ones in already-collected data that
#   can't be re-recorded); "exclude": left out of lab feedback.
# Overridden per check by ``confidence:`` / ``action:`` in the YAML. A check missing here defaults to
# DEFAULT_FEEDBACK, so a new check never reaches labs as a definite claim.
CONFIDENCE_LEVELS = ("high_confidence", "lower_confidence")
LAB_ACTIONS = ("fix_and_resubmit", "confirm_and_discuss", "exclude")
DEFAULT_FEEDBACK = ("lower_confidence", "confirm_and_discuss")
DEFAULT_LAB_FEEDBACK: dict[str, dict[str, tuple[str, str]]] = {
    "general": {
        "duplicate_pairs": ("high_confidence", "fix_and_resubmit"),
        "length_mismatch": ("high_confidence", "fix_and_resubmit"),
        "action_chunk_shape": ("high_confidence", "fix_and_resubmit"),
        "constant_action": ("high_confidence", "confirm_and_discuss"),
        "command_rate": ("lower_confidence", "confirm_and_discuss"),
        "required_state_keys": ("high_confidence", "fix_and_resubmit"),
        "placeholder_metadata": ("high_confidence", "fix_and_resubmit"),
        "action_copies_state": ("lower_confidence", "confirm_and_discuss"),
        "scale_consistency": ("lower_confidence", "exclude"),
        "tracking_residual": ("lower_confidence", "exclude"),
        "delta_detection": ("lower_confidence", "exclude"),
        "gripper_binary_present": ("high_confidence", "fix_and_resubmit"),
        "gripper_range": ("high_confidence", "fix_and_resubmit"),
        "gripper_command_range": ("lower_confidence", "confirm_and_discuss"),
    },
    "joint_position": {
        "delta_detection": ("high_confidence", "fix_and_resubmit"),
        "tracking": ("lower_confidence", "confirm_and_discuss"),
        "joint_names": ("lower_confidence", "confirm_and_discuss"),
        "dof": ("high_confidence", "fix_and_resubmit"),
        "radians": ("high_confidence", "fix_and_resubmit"),
        "step_jump": ("lower_confidence", "confirm_and_discuss"),
        "gripper_in_joints": ("high_confidence", "fix_and_resubmit"),
        "normalized_range": ("lower_confidence", "confirm_and_discuss"),
    },
    "joint_velocity": {
        "dof": ("high_confidence", "fix_and_resubmit"),
        "tracking": ("lower_confidence", "confirm_and_discuss"),
        "looks_like_position": ("high_confidence", "fix_and_resubmit"),
        "normalized_range": ("lower_confidence", "confirm_and_discuss"),
        "speed_limit": ("lower_confidence", "confirm_and_discuss"),
        "smoothness": ("lower_confidence", "exclude"),
    },
    "cartesian_position": {
        "biarm_ordering": ("lower_confidence", "confirm_and_discuss"),
        "delta_detection": ("high_confidence", "fix_and_resubmit"),
        "pose_tracking": ("lower_confidence", "confirm_and_discuss"),
        "position_units": ("high_confidence", "fix_and_resubmit"),
        "quaternion_shape": ("high_confidence", "fix_and_resubmit"),
        "quaternion_sign_flip": ("lower_confidence", "exclude"),
        "rotation_jump": ("lower_confidence", "confirm_and_discuss"),
        "normalized_range": ("lower_confidence", "confirm_and_discuss"),
        "quaternion_scalar_order": ("lower_confidence", "exclude"),
    },
    "cartesian_velocity": {
        "shape": ("high_confidence", "fix_and_resubmit"),
        "speed_limit": ("lower_confidence", "confirm_and_discuss"),
        "normalized_range": ("lower_confidence", "confirm_and_discuss"),
        "tracking": ("lower_confidence", "confirm_and_discuss"),
        "linear_angular_order": ("lower_confidence", "confirm_and_discuss"),
        "looks_like_position": ("high_confidence", "fix_and_resubmit"),
        "smoothness": ("lower_confidence", "exclude"),
    },
    "video": {
        "resolution": ("lower_confidence", "confirm_and_discuss"),
        "decoding": ("high_confidence", "fix_and_resubmit"),
        "brightness": ("lower_confidence", "exclude"),
        "fps": ("high_confidence", "confirm_and_discuss"),
        "duration": ("high_confidence", "confirm_and_discuss"),
        "camera_duration_mismatch": ("high_confidence", "confirm_and_discuss"),
        "corrupted_frames": ("lower_confidence", "confirm_and_discuss"),
        "video_jump": ("lower_confidence", "confirm_and_discuss"),
        "choppy": ("lower_confidence", "confirm_and_discuss"),
        "frozen": ("lower_confidence", "confirm_and_discuss"),
        "periodic_skips": ("lower_confidence", "confirm_and_discuss"),
        "recording_gap": ("lower_confidence", "confirm_and_discuss"),
        "state_jump": ("lower_confidence", "confirm_and_discuss"),
    },
}


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


# Category-level settings that aren't checks, with their defaults. Returned under
# CONFIG["<category>_settings"], e.g. CONFIG["video_settings"]["ignore_cameras"].
CATEGORY_SETTINGS: dict[str, dict] = {
    "video": {"ignore_cameras": [], "moving_fraction_of_p95": 0.1},
}


def _build_checks(raw: dict) -> tuple[dict, dict, dict]:
    """Returns ``(checks, lab_feedback, settings)``: per-check parameters, the per-check
    ``{"confidence", "action"}`` keyed ``"category.check"`` (kept apart so check functions
    never see it), and the category-level settings from CATEGORY_SETTINGS."""
    checks: dict[str, dict] = {}
    lab_feedback: dict[str, dict[str, str]] = {}
    settings: dict[str, dict] = {}
    for category, defaults in DEFAULT_CHECKS.items():
        raw_category = dict(raw.get(category) or {})
        # Optional category-level master switch: ``enabled: false`` turns off every check
        # in the category, whatever their own ``enabled`` says.
        category_enabled = bool(raw_category.pop("enabled", True))
        if category in CATEGORY_SETTINGS:
            settings[f"{category}_settings"] = {
                name: type(default)(raw_category.pop(name, default) or default)
                for name, default in CATEGORY_SETTINGS[category].items()
            }
        _reject_unknown(f"check(s) under {category!r}", raw_category, defaults)
        checks[category] = {}
        for check, params in defaults.items():
            raw_check = dict(raw_category.get(check) or {})
            default_conf, default_action = DEFAULT_LAB_FEEDBACK.get(category, {}).get(check, DEFAULT_FEEDBACK)
            feedback = {
                "confidence": str(raw_check.pop("confidence", default_conf)),
                "action": str(raw_check.pop("action", default_action)),
            }
            for field_name, allowed in (("confidence", CONFIDENCE_LEVELS), ("action", LAB_ACTIONS)):
                if feedback[field_name] not in allowed:
                    raise ValueError(
                        f"action_qa_config.yaml: {category}.{check}.{field_name} must be one of {allowed}, "
                        f"got {feedback[field_name]!r}"
                    )
            lab_feedback[f"{category}.{check}"] = feedback
            _reject_unknown(f"parameter(s) under {category}.{check}", raw_check, params)
            checks[category][check] = {
                name: type(default)(raw_check.get(name, default)) for name, default in params.items()
            }
            if not category_enabled:
                checks[category][check]["enabled"] = False
    return checks, lab_feedback, settings


def _build_config(raw: dict) -> dict:
    _reject_unknown("top-level section(s)", raw, [*DEFAULT_CHECKS, *_NON_CHECK_SECTIONS])
    sampling_raw = raw.get("sampling") or {}
    scope_raw = raw.get("scope") or {}
    run_raw = raw.get("run") or {}
    report_raw = raw.get("report") or {}
    _reject_unknown("parameter(s) under 'report'", report_raw, ["highlight_ignore"])
    highlight_ignore = [str(c) for c in (report_raw.get("highlight_ignore") or [])]
    _reject_unknown(
        "check(s) in report.highlight_ignore (use category.check)",
        highlight_ignore,
        [f"{cat}.{check}" for cat, checks in DEFAULT_CHECKS.items() for check in checks],
    )

    checks, lab_feedback, settings = _build_checks(raw)
    return {
        **checks,
        **settings,
        "lab_feedback": lab_feedback,
        "sampling": {
            "default_episode_limit": int(sampling_raw.get("default_episode_limit", 200)),
            # Of each dataset's sampled episodes, how many also get the (slow) video checks;
            # None = all of them.
            "video_episode_limit": (
                int(sampling_raw["video_episode_limit"]) if sampling_raw.get("video_episode_limit") is not None else None
            ),
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
        "report": {"highlight_ignore": highlight_ignore},
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
