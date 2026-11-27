"""Annotation columns of the episode index.

The ``annotation_*`` columns describe the annotation bundles: the kind of bundle
(``eval`` or ``train``) and the split an episode was assigned in, its current
holders, who returned an annotation, and the current collected annotations. A record holds the annotator's labels and segments,
the ``content_hash`` of the returned copy it was made on, and the annotator's language
instruction where it differs from the episode's.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from oopsie_backend.annotation.collect import current_annotations, read_store
from oopsie_backend.index.store import KEY, episode_key
from oopsie_backend.splits.eval_split import TRAIN_ANNOTATED

# Inclusive frame indices.
SEGMENT = pl.Struct(
    {
        "start": pl.Int64,
        "end": pl.Int64,
        "type": pl.String,
        "label": pl.String,
        "failure_category": pl.List(pl.String),
    }
)
ANNOTATION_RECORD = pl.Struct(
    {
        "annotator": pl.String,
        "name": pl.String,
        "timestamp": pl.String,
        "outcome": pl.String,
        "failure_category": pl.List(pl.String),
        "severity": pl.String,
        "episode_description": pl.String,
        "additional_notes": pl.String,
        "temporal_segments": pl.List(SEGMENT),
        "language_instruction": pl.String,
        "content_hash": pl.String,
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


def current_records(records: list[dict] | None, content_hash: str | None) -> tuple[list[dict], int]:
    """Each annotator's latest record, sorted by annotator, and the number of records left out
    because they were made on a copy whose ``content_hash`` differs from ``content_hash``."""
    latest: dict[str, dict] = {}
    stale = 0
    for r in records or []:
        if r["content_hash"] is not None and r["content_hash"] != content_hash:
            stale += 1
            continue
        if r["annotator"] not in latest or (r["timestamp"] or "") > (latest[r["annotator"]]["timestamp"] or ""):
            latest[r["annotator"]] = r
    return [latest[a] for a in sorted(latest)], stale


def _changed(instruction: str | None, original: str | None) -> str | None:
    return instruction if instruction and instruction != original else None


def _segments(record: dict) -> list[dict]:
    return [
        {
            "start": s.get("start"),
            "end": s.get("end"),
            "type": s.get("type"),
            "label": s.get("label"),
            "failure_category": s.get("failure_category") or [],
        }
        for s in json.loads(record["attrs"].get("temporal_segments") or "[]")
    ]


def annotation_frame(kinds: list[tuple[str, dict[str, dict], Path]], instructions: dict[str, str] | None = None) -> pl.DataFrame:
    """One row per assigned episode from ``(kind, assignment, store)`` triples.

    ``assignment`` is ``Registry.assignment(kind)``. ``instructions`` maps ``episode_key`` to the
    episode's language instruction; a record keeps the annotator's instruction only where it
    differs. An episode assigned in an eval split and in ``train_annotated`` keeps only its eval row.
    """
    instructions = instructions or {}
    rows = []
    for kind, assignment, store in kinds:
        current = current_annotations(read_store(store))
        for uid, meta in assignment.items():
            records = current.get(uid, [])
            key = episode_key(meta["dataset_repo"], meta["repo_path"])
            rows.append(
                {
                    KEY: key,
                    "annotation_batch": kind,
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
                            "additional_notes": r["annotation"].get("additional_notes"),
                            "temporal_segments": _segments(r),
                            "language_instruction": _changed(r.get("language_instruction"), instructions.get(key)),
                            "content_hash": r.get("content_hash"),
                        }
                        for r in records
                    ],
                }
            )
    df = pl.DataFrame(rows, schema=ANNOTATION_SCHEMA)
    df = df.filter(~(pl.col(KEY).is_duplicated() & (pl.col("annotation_split") == TRAIN_ANNOTATED)))
    dupes = df.filter(pl.col(KEY).is_duplicated())
    if dupes.height:
        raise ValueError(f"{dupes[KEY].n_unique()} episode(s) are assigned in more than one kind of bundle")
    return df
