"""Checks whether the video is too dark or too bright."""

from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("video", "brightness", VIDEO)
def brightness(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
