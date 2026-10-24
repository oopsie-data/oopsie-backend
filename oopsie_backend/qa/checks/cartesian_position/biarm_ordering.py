"""Checks whether the left and right arms are swapped in the actions (biarm only)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("cartesian_position", "biarm_ordering", ACTION)
def biarm_ordering(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
