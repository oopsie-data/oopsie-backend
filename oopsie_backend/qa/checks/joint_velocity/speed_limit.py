"""Checks whether any joint is commanded faster than a real arm can move."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_velocity", "speed_limit", ACTION)
def speed_limit(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
