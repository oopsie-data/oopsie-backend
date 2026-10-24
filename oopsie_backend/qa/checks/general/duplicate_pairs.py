"""Checks whether two different action types hold the same values (a likely copy-paste bug)."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "duplicate_pairs", ACTION)
def duplicate_pairs(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
