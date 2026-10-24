"""
Checks that the states the action space needs are recorded (gripper_position always; joint/cartesian position for joint/cartesian actions).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "required_state_keys", ACTION)
def required_state_keys(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
