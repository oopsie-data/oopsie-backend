"""Resolve recipe output formats to lazily imported writer functions.

Writers accept prepared episodes, output path, included manifest rows, layout
and recipe. HDF5 uses the prepared episodes directly and needs no writer.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

WRITERS: dict[str, str | None] = {
    "hdf5": None,
    "lerobot": "oopsie_backend.training.writers.lerobot",
    "rlds": "oopsie_backend.training.writers.rlds",
}


def writer(name: str) -> Callable | None:
    module = WRITERS[name]
    return importlib.import_module(module).write if module else None
