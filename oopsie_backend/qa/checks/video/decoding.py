"""Checks whether the video file exists and decodes without errors."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "decoding", VIDEO)
def decoding(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
