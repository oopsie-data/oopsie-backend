"""Every pipeline stage module, imported here by name so its ``@stage`` functions register."""

from __future__ import annotations

import oopsie_backend.annotation.stages  # noqa: F401
import oopsie_backend.index.stages  # noqa: F401
import oopsie_backend.postqa.stages  # noqa: F401
import oopsie_backend.preqa.stages  # noqa: F401
import oopsie_backend.qa.stages  # noqa: F401
import oopsie_backend.splits.stages  # noqa: F401
import oopsie_backend.stats.stages  # noqa: F401
from oopsie_backend.pipeline.registry import STAGES

# Named targets of ``oopsie pipeline run``, each a list of stages to compute.
TARGETS: dict[str, list[str]] = {
    "index": ["episode_index"],
    "preqa": ["duplicate_episodes", "duplicate_videos"],
    "qa": ["qa_results"],
    "postqa": ["critical_episodes"],
    "stats": ["stats"],
    "splits": ["eval_split"],
    "annotation": ["annotation_columns"],
    "nightly": ["stats"],
}

__all__ = ["STAGES", "TARGETS"]
