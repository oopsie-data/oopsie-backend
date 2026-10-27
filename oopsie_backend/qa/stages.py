"""Pipeline stages of the QA checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, stage
from oopsie_backend.qa import run
from oopsie_backend.qa.action_qa_config import CONFIG
from oopsie_backend.qa.checks.registry import enabled_checks
from oopsie_backend.qa.columns import QA_COLUMNS


@stage(always_run=True)
def qa_check_versions() -> str:
    """Hash of every enabled check's version and the sampling settings."""
    versions = {n: c.version(CONFIG) for n, c in enabled_checks(CONFIG).items()}
    payload = {"versions": versions, "sampling": CONFIG["sampling"], "scope": CONFIG["scope"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@stage
def qa_results(
    duplicate_episodes: str,
    qa_check_versions: str,
    data_dir: str,
    index_path: str,
    workers: int,
    reports_dir: str,
    node_stage_dir: str | None,
) -> str:
    """Runs every pending check on the QA sample; returns the fingerprint of the QA columns."""
    argv = ["--data-dir", data_dir, "--index", index_path, "--workers", str(workers)]
    argv += ["--stage-dir", node_stage_dir] if node_stage_dir else []
    check_rc(run.main(argv + ["--output", str(Path(reports_dir) / "action_qa.html")]), "QA")
    return columns_fingerprint(Path(index_path), QA_COLUMNS)
