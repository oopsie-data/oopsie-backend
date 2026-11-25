"""Load and validate training recipes, defaults and camera presets.

Defines selection, conversion and output settings, and fingerprints the settings
and implementation files that determine whether prepared episodes can be reused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from oopsie_backend.common.repos import CONTRIBUTOR, EXTERNAL

FORMATS = ("hdf5", "lerobot", "rlds")
ACTION_MODES = ("target", "native")
OUTCOMES = ("all", "success", "failure", "annotated")
# ``train`` is every episode the eval split does not hold out; the others are ``eval_split`` values.
SPLITS = ("train", "id", "od_task_id_scene", "od_task_od_scene", "excluded")
GRIPPER_MODES = ("normalized", "raw")
UPSAMPLE_MODES = ("interpolate", "hold", "native")
FROM_STATES_MODES = ("never", "missing", "always")
ARM_KEYS = ("joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity")
GRIPPER_KEYS = ("gripper_position", "gripper_velocity", "gripper_binary")
BASE_KEYS = ("base_position", "base_velocity")
ACTION_KEYS = ARM_KEYS + GRIPPER_KEYS + BASE_KEYS
STATE_KEYS = ("joint_position", "cartesian_position", "gripper_position")
# The state each action key can be generated from.
STATE_SOURCE = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_position",
    "cartesian_position": "cartesian_position",
    "cartesian_velocity": "cartesian_position",
    "gripper_position": "gripper_position",
    "gripper_binary": "gripper_position",
}
# Settings whose keys are names chosen by the recipe, not fixed settings.
FREE_FORM = {"pad_to", "roles", "fallback"}

_EXTERIOR_2 = ["exterior_front_2", "exterior_side_2", "exterior_top_2", "exterior_back_2"]
# Camera roles a recipe can name with ``cameras.preset`` instead of listing them.
# two_exterior_two_wrist: up to two wrist and two scene cameras, each optional. The first exterior
# role prefers the most informative placement, the second a different placement,
# so two cameras show the scene from two sides where an episode has them.
# Stereo images are never used, as they repeat their first image.
CAMERA_PRESETS = {
    "two_exterior_two_wrist": {
        "roles": {
            "wrist_1": ["wrist", "wrist_left"],
            "wrist_2": ["wrist_right", "wrist_2"],
            "exterior_1": ["exterior_front", "exterior_side", "exterior_top", "exterior_back", *_EXTERIOR_2, "head"],
            "exterior_2": ["exterior_side", "exterior_top", "exterior_front", "exterior_back", *_EXTERIOR_2, "head"],
        },
        "optional": ["wrist_1", "wrist_2", "exterior_1", "exterior_2"],
    },
}

DEFAULTS: dict = {
    "formats": ["hdf5"],
    "selection": {
        "repo_kinds": [CONTRIBUTOR, EXTERNAL],
        "splits": ["train"],
        "outcome": "all",
        "datasets": None,
        "exclude_datasets": [],
        "labs": None,
        "exclude_labs": [],
        "robots": None,
        "biarm": None,
        "binary_gripper": None,
        "dexterous_hand": None,
        "min_steps": 2,
        "min_seconds": 5,
        "max_episodes": None,
        "seed": 0,
    },
    "rate_hz": 15,
    "upsample": "interpolate",
    "actions": {
        "mode": "target", "keys": None, "orientation": "quat", "gripper": "normalized",
        "from_states": "never", "pad_to": {},
    },
    "state": {"keys": None, "pad_to": {}},
    "cameras": {
        "image_size": [180, 320], "preset": None, "roles": {}, "optional": [], "fallback": {}, "missing": "skip", "crf": 23,
    },
}


def _merge(defaults: dict, given: dict, where: str) -> dict:
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"{where}: unknown settings {sorted(unknown)}")
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        nested = isinstance(defaults[key], dict) and key not in FREE_FORM
        out[key] = _merge(defaults[key], value or {}, f"{where}.{key}") if nested else value
    return out


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def has_action_or_state(recipe: dict) -> bool:
    return recipe["actions"]["mode"] == "native" or bool(recipe["actions"]["keys"] or recipe["state"]["keys"])


def load_recipe(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    name = raw.pop("name", None)
    _check(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"{path}: name must be snake_case")
    recipe = {"name": name, **_merge(DEFAULTS, raw, str(path))}
    sel, act, state, cams = recipe["selection"], recipe["actions"], recipe["state"], recipe["cameras"]
    _check(set(recipe["formats"]) <= set(FORMATS) and recipe["formats"], f"{path}: formats must be from {FORMATS}")
    _check(sel["outcome"] in OUTCOMES, f"{path}: selection.outcome must be one of {OUTCOMES}")
    _check(bool(sel["splits"]) and set(sel["splits"]) <= set(SPLITS), f"{path}: selection.splits must be from {SPLITS}")
    _check(act["mode"] in ACTION_MODES, f"{path}: actions.mode must be one of {ACTION_MODES}")
    _check(act["gripper"] in GRIPPER_MODES, f"{path}: actions.gripper must be one of {GRIPPER_MODES}")
    _check(recipe["upsample"] in UPSAMPLE_MODES, f"{path}: upsample must be one of {UPSAMPLE_MODES}")
    _check(act["from_states"] in FROM_STATES_MODES, f"{path}: actions.from_states must be one of {FROM_STATES_MODES}")
    if act["mode"] == "target":
        _check(set(act["keys"] or []) <= set(ACTION_KEYS), f"{path}: actions.keys must be from {ACTION_KEYS}")
        _check(set(state["keys"] or []) <= set(STATE_KEYS), f"{path}: state.keys must be from {STATE_KEYS}")
        _check("lerobot" not in recipe["formats"] or bool(act["keys"] and state["keys"]),
               f"{path}: LeRobot needs both action and state keys in target mode")
        _check(bool(act["keys"]) or (not act["pad_to"] and act["from_states"] == "never"),
               f"{path}: actions.pad_to and actions.from_states require action keys")
        _check(bool(state["keys"]) or not state["pad_to"], f"{path}: state.pad_to requires state keys")
    else:
        _check(not act["keys"] and not state["keys"], f"{path}: native mode keeps every key; leave actions.keys and state.keys unset")
        _check(act["from_states"] == "never", f"{path}: actions.from_states needs target mode")
    if act["from_states"] == "always":
        _check(set(act["keys"]) <= set(STATE_SOURCE), f"{path}: actions.from_states: always needs actions.keys from {sorted(STATE_SOURCE)}")
    if cams["preset"] is not None:
        _check(cams["preset"] in CAMERA_PRESETS, f"{path}: cameras.preset must be one of {sorted(CAMERA_PRESETS)}")
        _check(not cams["roles"] and not cams["optional"], f"{path}: cameras.preset replaces cameras.roles and cameras.optional")
        cams.update(copy.deepcopy(CAMERA_PRESETS[cams["preset"]]))
    _check(bool(cams["roles"]), f"{path}: cameras.roles must name at least one camera role")
    _check(set(cams["optional"]) <= set(cams["roles"]), f"{path}: cameras.optional must name roles")
    _check(cams["missing"] in ("skip", "blank"), f"{path}: cameras.missing must be skip or blank")
    _check(set(cams["fallback"]) <= set(cams["roles"]) and set(cams["fallback"].values()) <= set(cams["roles"]),
           f"{path}: cameras.fallback must map roles to roles")
    if cams["image_size"] is None:
        _check(recipe["formats"] == ["hdf5"], f"{path}: cameras.image_size: null keeps source resolutions, which only hdf5 can hold")
    else:
        _check(len(cams["image_size"]) == 2 and min(cams["image_size"]) > 0, f"{path}: cameras.image_size is [height, width]")
    return recipe


# Modules whose code shapes a unified episode.
UNIFY_CODE = ("read_episode.py", "convert_actions_states.py", "resample.py", "cameras.py", "prepare_episode.py")


def unify_hash(recipe: dict) -> str:
    """Hash of the settings and code that change a unified episode."""
    keys = ("rate_hz", "upsample", "actions", "state", "cameras")
    h = hashlib.sha256(json.dumps({k: recipe[k] for k in keys}, sort_keys=True).encode())
    for name in UNIFY_CODE:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]
