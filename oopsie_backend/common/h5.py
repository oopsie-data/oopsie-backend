"""Decoding helpers for Oopsie HDF5 attributes and groups."""

from __future__ import annotations

import json

import h5py
import numpy as np

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
