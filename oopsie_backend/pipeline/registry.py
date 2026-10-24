"""The ``@stage`` decorator and the registry of pipeline stages.

A stage is a function registered with ``@stage``. Hamilton builds the pipeline
from the registered functions only: a stage's parameters name the stages or
config inputs it depends on, so the dependency graph is the function signatures.
Stages write their results to the episode index (or report files) and return a
fingerprint of what they wrote; Hamilton caches that fingerprint and reruns a
stage only when its code or an upstream fingerprint changes.

``@stage(always_run=True)`` marks stages that observe outside state (files on
disk, config files); they run every time and their fingerprint decides whether
downstream stages rerun. ``pipeline/stages.py`` imports every stage module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

from hamilton.function_modifiers import cache

STAGES: dict[str, Callable] = {}


def stage(fn: Callable | None = None, *, always_run: bool = False):
    """Register ``fn`` as a pipeline stage named after the function."""

    def register(f: Callable) -> Callable:
        if f.__name__ in STAGES:
            raise ValueError(f"stage {f.__name__} registered twice")
        if always_run:
            f = cache(behavior="recompute")(f)
        STAGES[f.__name__] = f
        return f

    return register(fn) if fn is not None else register


def check_rc(rc: int, name: str) -> None:
    if rc:
        raise RuntimeError(f"{name} exited {rc}")


def files_fingerprint(paths: Iterable[Path]) -> str:
    """Hash of the contents of ``paths``; missing files hash as absent."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p).encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()[:16]
