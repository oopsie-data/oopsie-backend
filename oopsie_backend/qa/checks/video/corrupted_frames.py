"""Checks for single frames unlike both their neighbours (decoding or recording glitches)."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "corrupted_frames", VIDEO)
def corrupted_frames(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
