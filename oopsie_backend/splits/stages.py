"""Pipeline stage that records the chosen eval split in the index."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage
from oopsie_backend.splits import record
from oopsie_backend.splits.columns import EVAL_COLUMNS


@stage(always_run=True)
def eval_proposal(proposal_path: str) -> str:
    """Fingerprint of the chosen proposal file."""
    return files_fingerprint([Path(proposal_path)])


@stage
def eval_split(episode_index: str, eval_proposal: str, proposal_path: str, index_path: str) -> str:
    """``eval_split`` and ``eval_proposal``; returns their fingerprint."""
    check_rc(record.main(["--proposal", proposal_path, "--index", index_path]), "record eval split")
    return columns_fingerprint(Path(index_path), EVAL_COLUMNS)
