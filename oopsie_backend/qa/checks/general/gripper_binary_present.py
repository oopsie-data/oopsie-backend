"""Checks that a non-empty gripper_binary action is recorded."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "gripper_binary_present", ACTION)
def gripper_binary_present(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
