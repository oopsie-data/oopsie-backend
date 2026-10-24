"""Checks whether operator, policy, robot or gripper names are blank or template placeholders."""

from oopsie_backend.qa.checks.base import METADATA, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("general", "placeholder_metadata", METADATA)
def placeholder_metadata(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
