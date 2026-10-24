"""Checks whether xyz actions peak at exactly +-1, i.e. were normalized instead of metres."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "normalized_range", ACTION)
def normalized_range(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
