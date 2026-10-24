"""Checks that positions are in metres (catches millimetres/centimetres or a far-away frame)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "position_units", ACTION)
def position_units(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
