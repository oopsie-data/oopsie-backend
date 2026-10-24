"""Checks whether the video stops changing for a while as the robot moves."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "frozen", VIDEO)
def frozen(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
