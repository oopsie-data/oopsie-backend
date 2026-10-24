"""Checks whether velocities flip direction from step to step (noise or misaligned timestamps)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_velocity", "smoothness", ACTION)
def smoothness(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
