"""
Checks whether the robot arm barely moves over the whole episode (an idle or failed recording).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "static_robot", ACTION)
def static_robot(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
