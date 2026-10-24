"""The common layout of a materialized dataset: key order, padded widths and masks.

``action`` and ``state`` vectors concatenate their keys in layout order, each
padded with zeros to the largest width any included episode has (or the recipe's
``pad_to`` width). The masks mark which entries an episode actually has.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import polars as pl

from oopsie_backend.training.recipe import ACTION_KEYS, STATE_KEYS


def build_layout(recipe: dict, included: pl.DataFrame) -> dict:
    layout = {"roles": list(recipe["cameras"]["roles"]), "image_size": list(recipe["cameras"]["image_size"])}
    for part, order, cfg in (("action", ACTION_KEYS, recipe["actions"]), ("state", STATE_KEYS, recipe["state"])):
        widths: dict[str, int] = {}
        for raw in included[f"{part}_widths"].to_list():
            for k, w in json.loads(raw).items():
                widths[k] = max(widths.get(k, 0), w)
        keys = cfg["keys"] or [k for k in order if k in widths]
        for k, w in (cfg["pad_to"] or {}).items():
            if w < widths.get(k, 0):
                raise ValueError(f"{part}s.pad_to.{k} = {w} is smaller than the widest episode ({widths[k]})")
            widths[k] = w
        layout[part] = {k: widths.get(k, 0) for k in keys}
    fps = included["fps"].unique().to_list()
    layout["fps"] = fps[0] if len(fps) == 1 else None
    return layout


def dim_names(spec: dict[str, int]) -> list[str]:
    return [f"{k}_{i}" for k, w in spec.items() for i in range(w)]


def padded(group: h5py.Group, spec: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """``(vector, mask, per_key)`` for a unified episode's ``action`` or ``state`` group."""
    length = next((group[k].shape[0] for k in group), 0)
    blocks, masks, per_key = [], [], {}
    for k, w in spec.items():
        block = np.zeros((length, w), dtype=np.float32)
        mask = np.zeros(w, dtype=bool)
        if k in group:
            value = np.asarray(group[k], dtype=np.float32)
            block[:, : value.shape[1]] = value
            mask[: value.shape[1]] = True
        blocks.append(block)
        masks.append(mask)
        per_key[k] = block
    vector = np.concatenate(blocks, axis=1) if blocks else np.zeros((length, 0), dtype=np.float32)
    return vector, np.concatenate(masks) if masks else np.zeros(0, dtype=bool), per_key
