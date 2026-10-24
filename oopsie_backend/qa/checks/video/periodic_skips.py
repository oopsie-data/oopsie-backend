"""
Checks for recording skips that drop time from state and video at a regular period (from the robot state).
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "periodic_skips", VIDEO)
def periodic_skips(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
