"""Checks whether the video resolution is high enough."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "resolution", VIDEO)
def resolution(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
