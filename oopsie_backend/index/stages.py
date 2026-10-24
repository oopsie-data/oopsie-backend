"""Pipeline stages of the index: observe the mirror and build the core columns."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oopsie_backend.common.repos import IMPORTS_CONFIG, REPOSITORIES_CONFIG
from oopsie_backend.index import build
from oopsie_backend.index.core import CORE_SCHEMA, discover
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage


@stage(always_run=True)
def mirror_state(hub_mirror: str, data_dir: str, index_path: str) -> str:
    """Runs after the Hub sync. Fingerprint of every indexed h5's path, size and modification time, the repo and import configs and the index's existence."""
    h = hashlib.sha256(files_fingerprint([REPOSITORIES_CONFIG, IMPORTS_CONFIG]).encode())
    h.update(b"index" if Path(index_path).is_file() else b"no index")
    for dataset_repo, repo_path, path, _ in discover(Path(data_dir)):
        st = path.stat()
        h.update(f"{dataset_repo}/{repo_path}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:16]


@stage
def episode_index(mirror_state: str, data_dir: str, index_path: str, workers: int, use_hub: bool) -> str:
    """Core columns of the index; returns their fingerprint."""
    argv = ["--data-dir", data_dir, "--index", index_path, "--workers", str(workers)]
    check_rc(build.main(argv + ([] if use_hub else ["--no-hub"])), "index build")
    return columns_fingerprint(Path(index_path), list(CORE_SCHEMA))
