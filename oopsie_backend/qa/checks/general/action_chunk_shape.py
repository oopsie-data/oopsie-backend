"""
Checks that each step holds one action, not a policy's whole action chunk (an arm action - joint/cartesian position or velocity - shaped (T, chunk, dof)).
"""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "action_chunk_shape", ACTION)
def action_chunk_shape(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
