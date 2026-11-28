"""Process pools for code that runs inside the threaded pipeline driver."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def process_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """A pool whose workers start in a fresh interpreter.

    Forking a process that runs other threads can copy a lock one of them holds, which then never
    releases in the child. Worker functions and their arguments must therefore be picklable.
    """
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))
