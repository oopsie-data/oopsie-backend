"""Checks whether an action and its matching robot state have the same number of steps."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "length_mismatch", ACTION)
def length_mismatch(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
