"""Pipeline stages of the annotation batches."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.annotation import pipeline
from oopsie_backend.annotation.columns import ANNOTATION_COLUMNS
from oopsie_backend.annotation.hub import load_config
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage


@stage(always_run=True)
def annotation_state(annotation_config_path: str, data_dir: str) -> str:
    """Fingerprint of the annotation config, every assignment list and every annotation store."""
    cfg = load_config(Path(annotation_config_path), Path(data_dir))
    paths = [Path(annotation_config_path)]
    for b in cfg.batches:
        paths += sorted(b.assignment_dir.glob("*.json")) + [b.store]
    return files_fingerprint(paths)


@stage
def annotation_columns(
    episode_index: str, annotation_state: str, annotation_config_path: str, data_dir: str, index_path: str
) -> str:
    """The ``annotation_*`` columns; returns their fingerprint."""
    argv = ["--config", annotation_config_path, "--data-dir", data_dir, "--index", index_path, "index"]
    check_rc(pipeline.main(argv), "annotation index")
    return columns_fingerprint(Path(index_path), ANNOTATION_COLUMNS)
