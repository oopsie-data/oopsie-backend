"""Checks that the number of joint velocities matches the number of joint names."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_velocity", "dof", ACTION)
def dof(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
