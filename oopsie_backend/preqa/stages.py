"""Pipeline stages run before QA."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, stage
from oopsie_backend.preqa import duplicate_episodes as de
from oopsie_backend.preqa import duplicate_videos as dv


@stage
def duplicate_episodes(episode_index: str, index_path: str) -> str:
    """Duplicate-episode columns; returns their fingerprint."""
    check_rc(de.main(["--index", index_path]), "duplicate episodes")
    return columns_fingerprint(Path(index_path), de.COLUMNS)


@stage
def duplicate_videos(duplicate_episodes: str, index_path: str) -> str:
    """Duplicate-video columns; returns their fingerprint."""
    check_rc(dv.main(["--index", index_path]), "duplicate videos")
    return columns_fingerprint(Path(index_path), dv.COLUMNS)
