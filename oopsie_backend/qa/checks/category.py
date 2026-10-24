"""Run a check through its category's shared logic, with every other check disabled."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from oopsie_backend.qa.action_qa import (
    profile_control_freq,
    run_action_qa_on_arrays,
    run_metadata_checks,
)
from oopsie_backend.qa.checks.base import ACTION, METADATA, VIDEO, Check
from oopsie_backend.qa.video_checks import run_video_checks

EMPTY = {"findings": [], "skips": [], "checked": []}


def only_config(config: dict, check: Check) -> dict:
    """``config`` with every check disabled except ``check``."""
    out = copy.deepcopy(config)
    for category, tests in out.items():
        if not isinstance(tests, dict):
            continue
        for test, cfg in tests.items():
            if isinstance(cfg, dict) and "enabled" in cfg:
                cfg["enabled"] = category == check.category and test == check.test
    return out


def run_category(ctx, config: dict, check: Check) -> dict:
    """``{"findings", "skips", "checked"}`` of ``check`` from its category's logic."""
    cfg = only_config(config, check)
    if check.kind == ACTION:
        return run_action_qa_on_arrays(ctx.actions, ctx.states, ctx.profile, cfg) or EMPTY
    if check.kind == METADATA:
        return run_metadata_checks(SimpleNamespace(attrs=ctx.attrs), ctx.profile, cfg)
    if check.kind == VIDEO:
        return run_video_checks(
            ctx.videos, ctx.states, profile_control_freq(ctx.profile), cfg["video"],
            cfg["video_settings"], ctx.actions, probe=ctx.probe,
        )
    raise ValueError(f"unknown check kind {check.kind!r}")
