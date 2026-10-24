"""
Checks whether an arm action never changes over the whole episode (stuck policy or placeholder actions).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "constant_action", ACTION)
def constant_action(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
