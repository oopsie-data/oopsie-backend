"""Training data recipes: which episodes, how they are unified, and the output formats.

A recipe is a YAML file in ``configs/training/``; ``load_recipe`` fills in the
defaults below and validates it. ``unify_hash`` covers every setting and the code that
change a unified episode, so unified episodes are reused across format changes.
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
GRIPPER_MODES = ("normalized", "raw")
ARM_KEYS = ("joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity")
GRIPPER_KEYS = ("gripper_position", "gripper_velocity", "gripper_binary")
BASE_KEYS = ("base_position", "base_velocity")
ACTION_KEYS = ARM_KEYS + GRIPPER_KEYS + BASE_KEYS
STATE_KEYS = ("joint_position", "cartesian_position", "gripper_position")
# Settings whose keys are names chosen by the recipe, not fixed settings.
FREE_FORM = {"pad_to", "roles", "fallback"}

DEFAULTS: dict = {
    "formats": ["hdf5"],
    "selection": {
        "repo_kinds": [CONTRIBUTOR, EXTERNAL],
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
        "max_episodes": None,
        "seed": 0,
    },
    "rate_hz": None,
    "actions": {"mode": "target", "keys": None, "orientation": "quat", "gripper": "normalized", "pad_to": {}},
    "state": {"keys": None, "pad_to": {}},
    "cameras": {"image_size": [180, 320], "roles": {}, "fallback": {}, "missing": "skip", "crf": 23},
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


def load_recipe(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    name = raw.pop("name", None)
    _check(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"{path}: name must be snake_case")
    recipe = {"name": name, **_merge(DEFAULTS, raw, str(path))}
    sel, act, state, cams = recipe["selection"], recipe["actions"], recipe["state"], recipe["cameras"]
    _check(set(recipe["formats"]) <= set(FORMATS) and recipe["formats"], f"{path}: formats must be from {FORMATS}")
    _check(sel["outcome"] in OUTCOMES, f"{path}: selection.outcome must be one of {OUTCOMES}")
    _check(act["mode"] in ACTION_MODES, f"{path}: actions.mode must be one of {ACTION_MODES}")
    _check(act["gripper"] in GRIPPER_MODES, f"{path}: actions.gripper must be one of {GRIPPER_MODES}")
    if act["mode"] == "target":
        _check(bool(act["keys"]) and set(act["keys"]) <= set(ACTION_KEYS), f"{path}: target mode needs actions.keys from {ACTION_KEYS}")
        _check(bool(state["keys"]) and set(state["keys"]) <= set(STATE_KEYS), f"{path}: target mode needs state.keys from {STATE_KEYS}")
    else:
        _check(not act["keys"] and not state["keys"], f"{path}: native mode keeps every key; leave actions.keys and state.keys unset")
    _check(bool(cams["roles"]), f"{path}: cameras.roles must name at least one camera role")
    _check(cams["missing"] in ("skip", "blank"), f"{path}: cameras.missing must be skip or blank")
    _check(set(cams["fallback"]) <= set(cams["roles"]) and set(cams["fallback"].values()) <= set(cams["roles"]),
           f"{path}: cameras.fallback must map roles to roles")
    _check(len(cams["image_size"]) == 2 and min(cams["image_size"]) > 0, f"{path}: cameras.image_size is [height, width]")
    return recipe


# Modules whose code shapes a unified episode.
UNIFY_CODE = ("source.py", "convert.py", "cameras.py", "unify.py")


def unify_hash(recipe: dict) -> str:
    """Hash of the settings and code that change a unified episode."""
    keys = ("rate_hz", "actions", "state", "cameras")
    h = hashlib.sha256(json.dumps({k: recipe[k] for k in keys}, sort_keys=True).encode())
    for name in UNIFY_CODE:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]
