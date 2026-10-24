"""Annotation columns of the episode index.

The ``annotation_*`` columns describe the annotation batches: which batch and
split an episode was assigned in, to whom, who returned an annotation, and the
current collected annotations.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from oopsie_backend.annotation.collect import (
    current_annotations,
    load_assignment,
    read_store,
)
from oopsie_backend.index.store import KEY, episode_key

ANNOTATION_RECORD = pl.Struct(
    {
        "annotator": pl.String,
        "name": pl.String,
        "timestamp": pl.String,
        "outcome": pl.String,
        "failure_category": pl.List(pl.String),
        "severity": pl.String,
        "episode_description": pl.String,
    }
)
ANNOTATION_SCHEMA = {
    KEY: pl.String,
    "annotation_batch": pl.String,
    "annotation_split": pl.String,
    "annotation_assigned": pl.List(pl.String),
    "annotation_returned": pl.List(pl.String),
    "annotation_records": pl.List(ANNOTATION_RECORD),
}
ANNOTATION_COLUMNS = [c for c in ANNOTATION_SCHEMA if c != KEY]


def annotation_frame(batches: list[tuple[str, Path, Path]]) -> pl.DataFrame:
    """One row per assigned episode from ``(batch, assignment_dir, store)`` triples."""
    rows = []
    for batch, assignment_dir, store in batches:
        assignment = load_assignment(assignment_dir)
        current = current_annotations(read_store(store))
        for uid, meta in assignment.items():
            records = current.get(uid, [])
            rows.append(
                {
                    KEY: episode_key(meta["dataset_repo"], meta["repo_path"]),
                    "annotation_batch": batch,
                    "annotation_split": meta["split"],
                    "annotation_assigned": sorted(meta["assigned"]),
                    "annotation_returned": sorted({r["bundle"] for r in records}),
                    "annotation_records": [
                        {
                            "annotator": r["bundle"],
                            "name": r["name"],
                            "timestamp": r.get("timestamp"),
                            "outcome": r["annotation"].get("outcome"),
                            "failure_category": r["annotation"].get("failure_category"),
                            "severity": r["annotation"].get("severity"),
                            "episode_description": r["annotation"].get("episode_description"),
                        }
                        for r in records
                    ],
                }
            )
    df = pl.DataFrame(rows, schema=ANNOTATION_SCHEMA)
    dupes = df.filter(pl.col(KEY).is_duplicated())
    if dupes.height:
        raise ValueError(f"{dupes[KEY].n_unique()} episode(s) are assigned in more than one batch")
    return df
