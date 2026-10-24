"""Checks whether the commanded orientation turns unrealistically fast between two steps."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "rotation_jump", ACTION)
def rotation_jump(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
