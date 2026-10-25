"""Progress logging for long loops: count, share, rate and estimated time remaining."""

from __future__ import annotations

import sys
import time


def _duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Progress:
    """Logs ``n/total <what>`` with rate and estimated remaining time at most every ``interval`` seconds."""

    def __init__(self, total: int, what: str, interval: float = 30.0):
        self.total, self.what, self.interval = total, what, interval
        self.start = self.last = time.monotonic()

    def tick(self, n: int) -> None:
        t = time.monotonic()
        if n < self.total and t - self.last < self.interval:
            return
        self.last = t
        elapsed = t - self.start
        rate = n / elapsed if elapsed else 0.0
        remaining = (self.total - n) / rate if rate else float("inf")
        print(
            f"  {n}/{self.total} {self.what} ({n / self.total:.0%}), {rate:.2f}/s, "
            f"elapsed {_duration(elapsed)}, remaining ~{_duration(remaining)}",
            file=sys.stderr,
        )
