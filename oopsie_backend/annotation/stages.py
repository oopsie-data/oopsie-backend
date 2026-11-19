"""Pipeline stages of the annotation batches: annotator assignments and the index columns.

Assignments are written once to ``<data_dir>/annotation/eval_assignment`` and
``train_assignment`` from the chosen proposal, the train sample and the settings in
``configs/eval_split.yaml``. The index columns hold every collected annotation, and the
flags of ``configs/annotation_flags.yaml`` mark the problems the annotators found;
pulling and collecting returns stay manual commands (``docs/annotation-coordinator.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from oopsie_backend.annotation import assign, flags, pipeline
from oopsie_backend.annotation.columns import ANNOTATION_COLUMNS
from oopsie_backend.annotation.hub import load_config
from oopsie_backend.common.repos import annotation_dir
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage, write_once
from oopsie_backend.splits.stages import NO_PROPOSAL, load_split_config

NO_ASSIGNMENTS = "no assignments"


def _assign_argv(proposal: Path, annotators: Path, cfg: dict, seed: int, out: Path) -> list[str]:
    argv = ["--proposal", str(proposal), "--annotators-file", str(annotators), "--out-dir", str(out)]
    argv += ["--copies", str(cfg.get("copies", 1)), "--seed", str(seed)]
    for split, n in (cfg.get("split_copies") or {}).items():
        argv += ["--split-copies", f"{split}={n}"]
    return argv


@stage
def annotator_assignments(
    chosen_proposal: str, train_sample: str, split_config: str, split_config_path: str, data_dir: str
) -> str:
    """``eval_assignment/`` and ``train_assignment/``, each written once; returns their fingerprint.

    Nothing is assigned until a proposal is chosen and ``assignments.annotators_file`` exists.
    """
    ann = annotation_dir(Path(data_dir))
    cfg = load_split_config(split_config_path).get("assignments", {})
    annotators = ann / cfg.get("annotators_file", "annotators.txt")
    if chosen_proposal == NO_PROPOSAL:
        return NO_ASSIGNMENTS
    if not annotators.is_file():
        print(f"waiting for annotators: write one name per line to {annotators}", file=sys.stderr)
        return NO_ASSIGNMENTS
    seed = cfg.get("seed", 0)
    fingerprints = []
    for name, source, upstream in (
        ("eval", ann / "eval_proposal.json", chosen_proposal),
        ("train", ann / "train_annotation_sample.json", train_sample),
    ):
        if not source.is_file():
            continue
        out = ann / f"{name}_assignment"
        batch_cfg = cfg.get(name, {})
        inputs = json.dumps(
            {"source": upstream, "annotators": files_fingerprint([annotators]), "settings": batch_cfg, "seed": seed},
            sort_keys=True,
        )
        argv = _assign_argv(source, annotators, batch_cfg, seed, out)
        fingerprints.append(write_once(out, inputs, lambda argv=argv, name=name: check_rc(assign.main(argv), f"assign {name}")))
    return "+".join(fingerprints) or NO_ASSIGNMENTS


@stage(always_run=True)
def annotation_state(annotation_config_path: str, data_dir: str) -> str:
    """Fingerprint of the annotation config and every annotation store."""
    cfg = load_config(Path(annotation_config_path), Path(data_dir))
    return files_fingerprint([Path(annotation_config_path), *(b.store for b in cfg.batches)])


@stage
def annotation_columns(
    annotator_assignments: str, annotation_state: str, annotation_config_path: str, data_dir: str, index_path: str
) -> str:
    """The ``annotation_*`` columns; returns their fingerprint.

    Without assignments the index is left unchanged.
    """
    if annotator_assignments == NO_ASSIGNMENTS:
        return NO_ASSIGNMENTS
    argv = ["--config", annotation_config_path, "--data-dir", data_dir, "--index", index_path, "index"]
    check_rc(pipeline.main(argv), "annotation index")
    return columns_fingerprint(Path(index_path), ANNOTATION_COLUMNS)



@stage(always_run=True)
def annotation_flags_config() -> str:
    """Fingerprint of ``configs/annotation_flags.yaml``."""
    return files_fingerprint([flags.FLAGS_CONFIG])


@stage
def annotation_flags(
    annotation_columns: str, episode_index: str, annotation_flags_config: str, data_dir: str, index_path: str, workers: int
) -> str:
    """``annotation_flags`` and ``annotation_excluded``; returns their fingerprint.

    Without assignments the index is left unchanged.
    """
    if annotation_columns == NO_ASSIGNMENTS:
        return NO_ASSIGNMENTS
    argv = ["--data-dir", data_dir, "--index", index_path, "--workers", str(workers)]
    check_rc(flags.main(argv), "annotation flags")
    return columns_fingerprint(Path(index_path), flags.COLUMNS)
