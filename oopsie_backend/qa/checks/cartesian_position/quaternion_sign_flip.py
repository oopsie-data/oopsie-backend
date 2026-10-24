"""
Checks whether the quaternion suddenly flips sign (q -> -q), same orientation but confusing for learning.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "quaternion_sign_flip", ACTION)
def quaternion_sign_flip(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
