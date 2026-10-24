"""
Guesses whether qw was put first instead of last (rough heuristic, often unreliable - off by default).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "quaternion_scalar_order", ACTION)
def quaternion_scalar_order(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
