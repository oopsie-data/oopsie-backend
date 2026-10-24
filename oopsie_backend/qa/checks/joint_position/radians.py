"""
Checks that joint values are in radians (catches degrees and the -100..100 range some arms use).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "radians", ACTION)
def radians(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
