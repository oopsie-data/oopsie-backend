"""
Checks for sudden image changes while the robot state changes smoothly (a cut, dropped frames, or video out of step).
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "video_jump", VIDEO, extra_sections=('corrupted_frames',))
def video_jump(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
