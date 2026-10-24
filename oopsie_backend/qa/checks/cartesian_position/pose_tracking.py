"""
Checks whether the commanded pose (position and orientation) matches where the robot actually ends up; flags if either is off.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "pose_tracking", ACTION)
def pose_tracking(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
