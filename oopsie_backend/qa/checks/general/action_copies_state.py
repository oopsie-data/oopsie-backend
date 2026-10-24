"""
Checks whether arm actions are just copies of the recorded state (rebuilt afterwards, not real commands).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "action_copies_state", ACTION)
def action_copies_state(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
