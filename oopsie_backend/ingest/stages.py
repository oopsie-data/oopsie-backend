"""Pipeline stage of the Hub mirror: bring ``<data_dir>/<org>/`` up to date with the Hub."""

from __future__ import annotations

import sys
from pathlib import Path

from oopsie_backend.common.repos import data_root
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.ingest import sync, upload_times
from oopsie_backend.pipeline.registry import files_fingerprint, stage


@stage(always_run=True)
def hub_mirror(data_dir: str, use_hub: bool) -> str:
    """Downloads new and changed files of every org repo; returns the fingerprint of the sync record.

    Repos that fail are listed in ``.sync_report.json`` and do not stop the pipeline.
    """
    if use_hub and sync.main(["--data-dir", data_dir]):
        print("hub_mirror: some repos failed to sync; see .sync_report.json", file=sys.stderr)
    return files_fingerprint([data_root(Path(data_dir)) / sync.STATE_NAME])


@stage
def hub_upload_times(hub_mirror: str, episode_index: str, data_dir: str, index_path: str, use_hub: bool) -> str:
    """``hub_uploaded_at`` and ``hub_upload_commit``; returns their fingerprint. No stage depends on them.

    Repos that fail keep their earlier values and do not stop the pipeline.
    """
    if use_hub and upload_times.main(["--data-dir", data_dir, "--index", index_path]):
        print("hub_upload_times: some repos failed; see the log above", file=sys.stderr)
    return columns_fingerprint(Path(index_path), upload_times.COLUMNS)
