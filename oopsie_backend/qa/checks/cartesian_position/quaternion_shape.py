"""
Checks that each arm's pose is [x, y, z, qx, qy, qz, qw] with a valid (unit-length) quaternion.
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "quaternion_shape", ACTION)
def quaternion_shape(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
