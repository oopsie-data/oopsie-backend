"""
Checks that velocity x step time roughly matches how far the end effector actually moved and turned.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_velocity", "tracking", ACTION)
def tracking(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
