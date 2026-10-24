"""Checks that the number of joint values matches the number of joint names (action and state)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "dof", ACTION)
def dof(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
