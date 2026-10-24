"""Checks whether the "velocity" is really a position (saved under the wrong key)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_velocity", "looks_like_position", ACTION)
def looks_like_position(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
