"""Checks whether velocities peak at exactly +-1, i.e. were normalized instead of real units."""

from oopsie_backend.qa.checks.base import ACTION, Check, check
from oopsie_backend.qa.checks.category import run_category


@check("joint_velocity", "normalized_range", ACTION)
def normalized_range(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
