"""Checks whether the video repeats frames while the robot moves."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "choppy", VIDEO)
def choppy(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
