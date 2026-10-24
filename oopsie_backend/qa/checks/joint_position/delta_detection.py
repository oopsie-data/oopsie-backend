"""Checks whether joint actions were saved as step-to-step changes instead of absolute positions."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "delta_detection", ACTION)
def delta_detection(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
