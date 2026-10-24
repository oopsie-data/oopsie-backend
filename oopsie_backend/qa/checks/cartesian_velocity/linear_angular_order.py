"""Checks whether the linear and angular halves are swapped."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_velocity", "linear_angular_order", ACTION, extra_sections=('tracking',))
def linear_angular_order(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
