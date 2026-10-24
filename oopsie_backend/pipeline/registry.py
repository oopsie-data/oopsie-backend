"""The ``@stage`` decorator and the registry of pipeline stages.

A stage is a function registered with ``@stage``. Hamilton builds the pipeline
from the registered functions only: a stage's parameters name the stages or
config inputs it depends on, so the dependency graph is the function signatures.
Stages write their results to the episode index (or report files) and return a
fingerprint of what they wrote; Hamilton caches that fingerprint and reruns a
stage only when its code or an upstream fingerprint changes.

``@stage(always_run=True)`` marks stages that observe outside state (files on
disk, config files); they run every time and their fingerprint decides whether
downstream stages rerun. ``@stage(manual=True)`` marks stages that run only
when requested by name or target, never as part of ``all``.
``pipeline/stages.py`` imports every stage module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

from hamilton.function_modifiers import cache

STAGES: dict[str, Callable] = {}
MANUAL: set[str] = set()


def stage(fn: Callable | None = None, *, always_run: bool = False, manual: bool = False):
    """Register ``fn`` as a pipeline stage named after the function."""

    def register(f: Callable) -> Callable:
        if f.__name__ in STAGES:
            raise ValueError(f"stage {f.__name__} registered twice")
        if always_run:
            f = cache(behavior="recompute")(f)
        STAGES[f.__name__] = f
        if manual:
            MANUAL.add(f.__name__)
        return f

    return register(fn) if fn is not None else register


def check_rc(rc: int, name: str) -> None:
    if rc:
        raise RuntimeError(f"{name} exited {rc}")


def write_once(
    output: Path, inputs: str, create: Callable[[], None], fingerprint: Callable[[Iterable[Path]], str] | None = None
) -> str:
    """Create ``output`` (a file or folder) once; returns the fingerprint of its contents.

    ``inputs`` describes what the output is made from and is recorded next to it as
    ``<name>.inputs``. Later calls keep the output; if ``inputs`` changed, they fail instead
    of rewriting it. An output without a record is adopted as is. Delete the output and its
    record to regenerate it. ``fingerprint`` defaults to ``files_fingerprint``.
    """
    record = output.with_name(output.name + ".inputs")
    if output.exists():
        recorded = record.read_text() if record.is_file() else None
        if recorded is not None and recorded != inputs:
            raise RuntimeError(f"{output} was made from other inputs; delete it and {record.name} to regenerate it")
        if recorded is None:
            record.write_text(inputs)
    else:
        create()
        record.write_text(inputs)
    paths = sorted(p for p in output.rglob("*") if p.is_file()) if output.is_dir() else [output]
    return (fingerprint or files_fingerprint)(paths)


def stat_fingerprint(paths: Iterable[Path]) -> str:
    """Hash of the paths, sizes and modification times of ``paths``; for large outputs."""
    h = hashlib.sha256()
    for p in paths:
        st = p.stat() if p.exists() else None
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns};".encode() if st else f"{p}:<missing>;".encode())
    return h.hexdigest()[:16]


def files_fingerprint(paths: Iterable[Path]) -> str:
    """Hash of the contents of ``paths``; missing files hash as absent."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p).encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()[:16]
