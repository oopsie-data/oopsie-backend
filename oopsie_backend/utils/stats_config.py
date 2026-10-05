"""Load configs/stats_config.yaml: dataset/lab exclusion lists and the lab
requirement thresholds. See that file for what each field means.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _BACKEND_ROOT / "configs" / "stats_config.yaml"

# Used only if the YAML file is missing, so the package still imports.
_FALLBACK_RAW: dict = {
    "datasets": {
        "stats_excluded": [
            "OopsieData-Submissions/oopsiedata-v0.1",
            "OopsieData-Submissions/oopsiedata-converted-v0.1",
            "OopsieData-Submissions/oopsie-face-raw-archive-2026-08-12",
        ],
        "public_excluded": [
            "OopsieData-Submissions/SOAR_origin",
            "OopsieData-Submissions/AutoEval_origin",
            "OopsieData-Submissions/roboarena_upload",
            "OopsieData-Submissions/AI2_MOLMOACT2",
            "OopsieData-Submissions/Amazon-MobileManipulationTeam",
        ],
        "non_lab_infra": [
            "OopsieData-Submissions/Visualizer_Previews",
            "OopsieData-Submissions/WebVisualizer",
        ],
    },
    "labs": {
        "public_excluded": ["example_institution", "your_lab_id"],
        "placeholder_operator_names": [
            "unknown", "operator", "pending_operator", "n/a", "na", "tbd",
            "none", "test", "test_operator", "default", "admin", "unset",
        ],
    },
    "requirements": {
        "alpha_cutoff_date": "2026-07-01",
        "alpha": {"enabled": True, "min_qualifying_episodes_per_individual": 50},
        "beta": {
            "volume": {
                "enabled": True,
                "min_qualifying_episodes_per_individual": 750,
                "min_failures_per_individual": 350,
                "min_duration_hours_per_individual": 5.0,
            },
            "task_diversity": {"enabled": False},
            "per_task_cap": {"enabled": False},
        },
    },
    "action_qa": {
        "duplicate_pairs": {"enabled": True, "atol": 1.0e-9, "rtol": 0.0},
        "scale_consistency": {"enabled": True, "max_ratio": 5.0, "min_ratio": 0.2},
        "tracking_residual": {"enabled": True, "max_normalized_rms": 0.5},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "sampling": {"default_episode_limit": 50},
    },
}


def _build_config(raw: dict) -> dict:
    datasets = raw.get("datasets") or {}
    labs = raw.get("labs") or {}
    req = raw.get("requirements") or {}
    alpha_raw = req.get("alpha") or {}
    beta_raw = req.get("beta") or {}
    volume_raw = beta_raw.get("volume") or {}
    task_diversity_raw = beta_raw.get("task_diversity") or {}
    per_task_cap_raw = beta_raw.get("per_task_cap") or {}
    action_qa = raw.get("action_qa") or {}
    dup_raw = action_qa.get("duplicate_pairs") or {}
    scale_raw = action_qa.get("scale_consistency") or {}
    tracking_raw = action_qa.get("tracking_residual") or {}
    delta_raw = action_qa.get("delta_detection") or {}
    sampling_raw = action_qa.get("sampling") or {}

    return {
        "stats_excluded_datasets": frozenset(datasets.get("stats_excluded") or []),
        "public_excluded_datasets": frozenset(datasets.get("public_excluded") or []),
        "non_lab_infra_datasets": frozenset(datasets.get("non_lab_infra") or []),
        "public_excluded_labs": frozenset(labs.get("public_excluded") or []),
        "placeholder_operator_names": frozenset(
            str(s).strip().lower() for s in (labs.get("placeholder_operator_names") or [])
        ),
        "requirements": {
            "alpha_cutoff_date": req.get("alpha_cutoff_date", "2026-07-01"),
            "alpha": {
                "enabled": bool(alpha_raw.get("enabled", True)),
                "min_qualifying_episodes_per_individual": int(
                    alpha_raw.get("min_qualifying_episodes_per_individual", 50)
                ),
            },
            "beta": {
                "volume": {
                    "enabled": bool(volume_raw.get("enabled", True)),
                    "min_qualifying_episodes_per_individual": int(
                        volume_raw.get("min_qualifying_episodes_per_individual", 750)
                    ),
                    "min_failures_per_individual": int(
                        volume_raw.get("min_failures_per_individual", 350)
                    ),
                    "min_duration_hours_per_individual": float(
                        volume_raw.get("min_duration_hours_per_individual", 5.0)
                    ),
                },
                "task_diversity_enabled": bool(task_diversity_raw.get("enabled", False)),
                "per_task_cap_enabled": bool(per_task_cap_raw.get("enabled", False)),
            },
        },
        "action_qa": {
            "duplicate_pairs": {
                "enabled": bool(dup_raw.get("enabled", True)),
                "atol": float(dup_raw.get("atol", 1.0e-9)),
                "rtol": float(dup_raw.get("rtol", 0.0)),
            },
            "scale_consistency": {
                "enabled": bool(scale_raw.get("enabled", True)),
                "max_ratio": float(scale_raw.get("max_ratio", 5.0)),
                "min_ratio": float(scale_raw.get("min_ratio", 0.2)),
            },
            "tracking_residual": {
                "enabled": bool(tracking_raw.get("enabled", True)),
                "max_normalized_rms": float(tracking_raw.get("max_normalized_rms", 0.5)),
            },
            "delta_detection": {
                "enabled": bool(delta_raw.get("enabled", True)),
                "ratio_threshold": float(delta_raw.get("ratio_threshold", 0.3)),
            },
            "sampling": {
                "default_episode_limit": int(sampling_raw.get("default_episode_limit", 50)),
            },
        },
    }


def load_stats_config(path: Path | None = None) -> dict:
    """Load and validate configs/stats_config.yaml into a plain dict.

    Falls back to built-in defaults (matching the shipped YAML) if the file is
    missing, so importing this module never hard-fails.
    """
    path = path or DEFAULT_CONFIG_PATH
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = _FALLBACK_RAW
    return _build_config(raw)


# Loaded once at import time; existing code reads exclusion sets off of this via
# oopsie_backend.utils.episode_stats's module-level constants.
CONFIG: dict = load_stats_config()
