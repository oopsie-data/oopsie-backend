"""
Checks whether the robot state jumps in a single step while the robot is moving but the video doesn't (a state glitch, or state and video out of step).
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "state_jump", VIDEO, extra_sections=('recording_gap',))
def state_jump(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
