"""Pipeline stage of the public and internal stats."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.common.stats_config import DEFAULT_CONFIG_PATH
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage
from oopsie_backend.stats import compute_local


@stage(always_run=True)
def stats_config() -> str:
    """Fingerprint of ``configs/stats_config.yaml``."""
    return files_fingerprint([DEFAULT_CONFIG_PATH])


@stage
def stats(
    critical_episodes: str, stats_config: str, data_dir: str, index_path: str, stats_docs_output: str, reports_dir: str
) -> str:
    """Writes the public and internal stats; returns the fingerprint of the docs ``stats.json``."""
    reports = Path(reports_dir)
    argv = [
        "--data-dir", data_dir,
        "--index", index_path,
        "--docs-output", stats_docs_output,
        "--public-json", str(reports / "public_stats.json"),
        "--public-html", str(reports / "public_stats.html"),
        "--internal-output", str(reports / "internal_stats.json"),
    ]
    check_rc(compute_local.main(argv), "stats")
    return files_fingerprint([Path(stats_docs_output)])
