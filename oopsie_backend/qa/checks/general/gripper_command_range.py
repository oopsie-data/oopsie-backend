"""
Checks whether gripper position commands go far beyond the range the gripper state actually covers.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "gripper_command_range", ACTION)
def gripper_command_range(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
