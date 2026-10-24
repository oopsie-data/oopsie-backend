"""Checks that each arm's velocity is 6 values: 3 linear then 3 angular."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_velocity", "shape", ACTION)
def shape(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
