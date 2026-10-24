"""Checks whether the commanded joint positions match where the robot's joints actually end up."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "tracking", ACTION, extra_sections=("radians",))
def tracking(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
