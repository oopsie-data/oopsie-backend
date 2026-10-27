"""Pipeline stages of the index: observe the mirror, build the core columns and name the cameras."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oopsie_backend.common.repos import IMPORTS_CONFIG, REPOSITORIES_CONFIG
from oopsie_backend.index import build, camera_names
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


@stage(always_run=True)
def camera_names_config() -> str:
    """Fingerprint of ``configs/camera_names.yaml``."""
    return files_fingerprint([camera_names.CAMERA_NAMES_CONFIG])


@stage
def unified_camera_names(episode_index: str, duplicate_episodes: str, camera_names_config: str, index_path: str) -> str:
    """``unified_cameras`` and ``camera_name_issues``; returns their fingerprint."""
    check_rc(camera_names.main(["--index", index_path]), "camera names")
    return columns_fingerprint(Path(index_path), camera_names.COLUMNS)
