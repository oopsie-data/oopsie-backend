"""Pipeline stage of the imported datasets (``configs/imports.yaml``).

Manual: ``oopsie pipeline run imports`` converts local imports that are missing.
The index stage picks up the converted folders through the mirror, so the
nightly pipeline never runs a conversion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oopsie_backend.conversion.imports import import_datasets
from oopsie_backend.pipeline.registry import stage


@stage(always_run=True, manual=True)
def imported_datasets(data_dir: str, stage_dir: str | None) -> str:
    """Converts missing local imports; returns the fingerprint of every imported dataset.

    With ``stage_dir`` set, conversions are written there and moved to the imports folder when done.
    """
    fingerprints = import_datasets(Path(data_dir), work_dir=Path(stage_dir) / "imports" if stage_dir else None)
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16]
