"""
Checks whether the video's fps matches control_freq (a header value; playback speed, not frame alignment).
"""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "fps", VIDEO)
def fps(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
