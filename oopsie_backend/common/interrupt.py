"""Stopping a run on SIGINT or SIGTERM, letting stages save their progress first.

Signal handlers can only be installed from the main thread, so the entry point
calls ``install``: a stage's ``main`` run on its own, or the pipeline driver,
whose stages run in worker threads. A stage that can save partial work wraps it
in ``stoppable`` and polls ``stopping``. On the first signal the process exits
at once if no such stage is running, and otherwise after ``GRACE_SECONDS``; a
second signal exits at once.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

GRACE_SECONDS = 20.0

_stop = threading.Event()
_lock = threading.Lock()
_active = 0


def _exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _handle(signum, frame) -> None:
    code = 128 + signum
    if _stop.is_set() or not _active:
        print(f"stopped by {signal.Signals(signum).name}", file=sys.stderr)
        _exit(code)
    _stop.set()
    print(f"stopping on {signal.Signals(signum).name}; saving progress, exiting in {GRACE_SECONDS:.0f}s", file=sys.stderr)
    timer = threading.Timer(GRACE_SECONDS, _exit, (code,))
    timer.daemon = True
    timer.start()


def install() -> None:
    """Handle SIGINT and SIGTERM; does nothing outside the main thread."""
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, _handle)


def stopping() -> bool:
    return _stop.is_set()


@contextmanager
def stoppable() -> Iterator[None]:
    """Marks a stage that saves its progress when ``stopping`` turns true."""
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1
