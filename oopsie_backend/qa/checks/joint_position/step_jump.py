"""Checks whether a joint moves unrealistically far between two steps."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_position", "step_jump", ACTION, extra_sections=("radians",))
def step_jump(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
