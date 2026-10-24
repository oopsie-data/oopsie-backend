"""
Checks whether a gripper value is mixed into the joint vector (it belongs in the gripper action).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "gripper_in_joints", ACTION)
def gripper_in_joints(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
