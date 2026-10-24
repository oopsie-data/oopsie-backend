"""
Checks whether the commanded position matches where the robot actually ends up (gripper position only).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "tracking_residual", ACTION)
def tracking_residual(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
