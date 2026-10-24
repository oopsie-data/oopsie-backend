"""Run QA checks on episodes: each episode is read once, then every pending check runs on it."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from oopsie_backend.common.videos import episode_videos
from oopsie_backend.qa.action_qa import METADATA_FIELDS, load_action_state_arrays
from oopsie_backend.qa.checks.base import Check
from oopsie_backend.qa.checks.registry import all_checks
from oopsie_backend.qa.video_checks import VideoProbe, probe_video


class EpisodeContext:
    """One episode's arrays, profile and root attributes; videos are decoded on first use and shared."""

    def __init__(self, path: Path, repo_path: str):
        self.path = path
        self.repo_path = repo_path
        with h5py.File(path, "r") as f:
            self.actions, self.states, self.profile = load_action_state_arrays(f)
            self.attrs = {k: f.attrs[k] for k in METADATA_FIELDS if k in f.attrs}
        self._videos: dict[str, Path] | None = None
        self._probes: dict[Path, VideoProbe] = {}

    @property
    def videos(self) -> dict[str, Path]:
        if self._videos is None:
            self._videos = episode_videos(self.path, self.repo_path)
        return self._videos

    def probe(self, path: Path) -> VideoProbe:
        if path not in self._probes:
            self._probes[path] = probe_video(path)
        return self._probes[path]


def run_check(ctx: EpisodeContext, check: Check, config: dict) -> dict:
    """``{"findings", "skips", "checked"}`` of one check on one episode."""
    return check.fn(ctx, config, check)


def run_episode(job: tuple[str, Path, str, list[str], dict]) -> tuple[str, dict[str, dict], float]:
    """Run the named checks on one episode; returns ``(key, {check: result}, seconds)``.

    A result is ``{"findings", "skips", "ran", "error"}``; ``ran`` says whether the check
    applied to this episode. A failure to read the episode is an error for every check.
    """
    key, path, repo_path, names, config = job
    start = time.monotonic()
    checks = all_checks()
    try:
        ctx = EpisodeContext(path, repo_path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return key, {n: {"findings": [], "skips": [], "ran": False, "error": error} for n in names}, 0.0
    out: dict[str, dict] = {}
    for name in names:
        check = checks[name]
        try:
            result = run_check(ctx, check, config)
            ran = (check.category, check.test) in {tuple(c) for c in result["checked"]}
            out[name] = {"findings": result["findings"], "skips": result["skips"], "ran": ran, "error": None}
        except Exception as exc:
            out[name] = {"findings": [], "skips": [], "ran": False, "error": f"{type(exc).__name__}: {exc}"}
    return key, out, time.monotonic() - start


def now() -> datetime:
    return datetime.now(timezone.utc)


def to_builtin(value):
    """JSON default for numpy values in finding details."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
