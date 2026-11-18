"""Every pipeline stage module, imported here by name so its ``@stage`` functions register."""

from __future__ import annotations

import oopsie_backend.annotation.stages  # noqa: F401
import oopsie_backend.conversion.stages  # noqa: F401
import oopsie_backend.index.stages  # noqa: F401
import oopsie_backend.ingest.stages  # noqa: F401
import oopsie_backend.postqa.stages  # noqa: F401
import oopsie_backend.preqa.stages  # noqa: F401
import oopsie_backend.qa.stages  # noqa: F401
import oopsie_backend.splits.stages  # noqa: F401
import oopsie_backend.stats.stages  # noqa: F401
import oopsie_backend.training.pipeline_stages  # noqa: F401
import inspect

from oopsie_backend.pipeline.registry import MANUAL, STAGES

# Every non-manual stage no other non-manual stage depends on; computing them runs the whole pipeline.
_AUTOMATIC = {n: f for n, f in STAGES.items() if n not in MANUAL}
FINAL_STAGES = sorted(set(_AUTOMATIC) - {p for f in _AUTOMATIC.values() for p in inspect.signature(f).parameters})

# Named targets of ``oopsie pipeline run``, each a list of stages to compute.
TARGETS: dict[str, list[str]] = {
    "sync": ["hub_mirror"],
    "index": ["episode_index", "hub_upload_times"],
    "preqa": ["duplicate_episodes", "duplicate_videos", "unified_camera_names"],
    "qa": ["qa_results"],
    "postqa": ["critical_episodes"],
    "stats": ["stats"],
    "proposals": ["eval_proposals"],
    "splits": ["eval_split", "train_sample"],
    "annotation": ["annotation_columns"],
    "all": FINAL_STAGES,
    "imports": ["imported_datasets"],
    "training": ["training_data"],
    "release": ["data_release"],
}

__all__ = ["FINAL_STAGES", "STAGES", "TARGETS"]
