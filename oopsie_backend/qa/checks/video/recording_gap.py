"""
Checks whether the robot state and the video jump at the same moment (time missing from the recording); uses state_jump's settings to find jumps.
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "recording_gap", VIDEO, extra_sections=('state_jump',))
def recording_gap(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
