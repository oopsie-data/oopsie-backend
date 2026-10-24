"""Checks whether actions are on the same scale as the robot state (gripper position only)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "scale_consistency", ACTION)
def scale_consistency(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
