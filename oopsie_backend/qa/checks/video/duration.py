"""
Checks whether the video lasts as long as the trajectory (decoded frames / fps vs T / control_freq) - i.e. whether frame i is still step i.
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "duration", VIDEO)
def duration(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
