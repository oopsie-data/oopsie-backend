"""Checks whether the cameras of one episode have the same duration."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "camera_duration_mismatch", VIDEO)
def camera_duration_mismatch(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
