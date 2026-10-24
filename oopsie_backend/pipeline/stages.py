"""Every pipeline stage module, imported here by name so its ``@stage`` functions register."""

from __future__ import annotations

import oopsie_backend.annotation.stages  # noqa: F401
import oopsie_backend.index.stages  # noqa: F401
import oopsie_backend.postqa.stages  # noqa: F401
import oopsie_backend.preqa.stages  # noqa: F401
import oopsie_backend.qa.stages  # noqa: F401
import oopsie_backend.splits.stages  # noqa: F401
import oopsie_backend.stats.stages  # noqa: F401
import inspect

from oopsie_backend.pipeline.registry import STAGES

# Every stage no other stage depends on; computing them runs the whole pipeline.
FINAL_STAGES = sorted(
    set(STAGES) - {p for f in STAGES.values() for p in inspect.signature(f).parameters}
)

# Named targets of ``oopsie pipeline run``, each a list of stages to compute.
TARGETS: dict[str, list[str]] = {
    "index": ["episode_index"],
    "preqa": ["duplicate_episodes", "duplicate_videos"],
    "qa": ["qa_results"],
    "postqa": ["critical_episodes"],
    "stats": ["stats"],
    "proposals": ["eval_proposals"],
    "splits": ["eval_split", "train_sample"],
    "annotation": ["annotation_columns"],
    "all": FINAL_STAGES,
}

__all__ = ["FINAL_STAGES", "STAGES", "TARGETS"]
