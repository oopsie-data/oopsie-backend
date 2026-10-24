"""
Checks whether each new arm action is held for the same N > 1 steps all episode (commands arrive at control_freq / N, not control_freq).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "command_rate", ACTION)
def command_rate(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
