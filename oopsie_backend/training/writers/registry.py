"""Output formats of a materialized dataset, by the name recipes use in ``formats``.

A writer is ``write(episodes_dir, out, included, layout, recipe)``: it reads the
unified episodes of the ``included`` manifest rows and writes the format into
``out``. ``hdf5`` needs no writer: the unified episodes, ``layout.json`` and
``manifest.parquet`` are that format. Writers are imported on use, so RLDS
loads TensorFlow only when a recipe asks for it.
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
