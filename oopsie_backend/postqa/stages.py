"""Pipeline stages run after QA."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage
from oopsie_backend.postqa import critical


@stage(always_run=True)
def critical_config() -> str:
    """Fingerprint of ``configs/critical.yaml``."""
    return files_fingerprint([critical.CRITICAL_CONFIG])


@stage
def critical_episodes(qa_results: str, duplicate_videos: str, critical_config: str, index_path: str) -> str:
    """``critical`` and ``critical_reasons``; returns their fingerprint."""
    check_rc(critical.main(["--index", index_path]), "critical")
    return columns_fingerprint(Path(index_path), critical.COLUMNS)
