"""
Checks that the profile's action and state joint names are given, match each other and are in a sensible order.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "joint_names", ACTION)
def joint_names(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
