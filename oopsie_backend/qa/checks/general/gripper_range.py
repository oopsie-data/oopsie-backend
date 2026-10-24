"""Checks that gripper_binary actions are in [0, 1] (catches e.g. -1/+1)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "gripper_range", ACTION)
def gripper_range(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
