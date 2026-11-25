"""Register manual training stages and their pipeline dependencies.

Fingerprints the recipe and builds its dataset after index, QA, split, annotation
and camera-name dependencies are current. Nightly runs do not build training sets.
"""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import files_fingerprint, stage
from oopsie_backend.training.build_dataset import TRAINING_COLUMN, materialize


@stage(always_run=True, manual=True)
def training_recipe_state(training_recipe: str | None) -> str:
    """Fingerprint of the recipe file."""
    if not training_recipe:
        raise ValueError("set training_recipe, e.g. --set training_recipe=configs/training/<recipe>.yaml")
    return files_fingerprint([Path(training_recipe)])


@stage(manual=True)
def training_data(
    critical_episodes: str,
    eval_split: str,
    annotation_flags: str,
    unified_camera_names: str,
    training_recipe_state: str,
    training_recipe: str,
    data_dir: str,
    index_path: str,
    training_dir: str,
    training_work_dir: str | None,
    node_stage_dir: str | None,
    workers: int,
) -> str:
    """Materializes ``training_recipe``; returns the fingerprint of the ``training_sets`` column."""
    materialize(
        Path(training_recipe),
        Path(data_dir),
        Path(index_path),
        Path(training_dir),
        Path(training_work_dir or training_dir),
        workers,
        Path(node_stage_dir) if node_stage_dir else None,
    )
    return columns_fingerprint(Path(index_path), [TRAINING_COLUMN])
