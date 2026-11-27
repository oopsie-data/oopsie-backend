"""Pipeline stages of the annotation bundles: the bundles and roster, and the index columns.

The first run with a chosen proposal and ``<data_dir>/annotation/annotators.txt`` assigns the
eval split and the train sample to the listed annotators (``eval_assignment/``, ``train_assignment/``)
and turns every annotator's list into a bundle in ``bundles/`` (``annotation.roster``). Bundles are
final. Every later name in ``annotators.txt`` gets a new train bundle (``annotation.train_bundles``);
``oopsie annotation bundle`` adds further bundles and reassigns them. The index columns hold every
collected annotation, and the flags of ``configs/annotation_flags.yaml`` mark the problems the
annotators found; staging, publishing and collecting returns stay manual commands
(``docs/pipeline/annotation/index.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from oopsie_backend.annotation import assign, flags, pipeline
from oopsie_backend.annotation.collect import current_annotations, read_store
from oopsie_backend.annotation.columns import ANNOTATION_COLUMNS
from oopsie_backend.annotation.hub import load_config
from oopsie_backend.annotation.roster import EVAL, ROSTER_NAME, TRAIN, Registry, adopt
from oopsie_backend.annotation.train_bundles import TrainPool, add_train_bundle
from oopsie_backend.common.repos import annotation_dir
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage, write_once
from oopsie_backend.splits.stages import NO_PROPOSAL, load_split_config

NO_ASSIGNMENTS = "no assignments"
LEGACY = ((EVAL, "batch_1", "eval_assignment"), (TRAIN, "batch_2", "train_assignment"))


def _assign_argv(proposal: Path, annotators: Path, cfg: dict, seed: int, out: Path) -> list[str]:
    argv = ["--proposal", str(proposal), "--annotators-file", str(annotators), "--out-dir", str(out)]
    argv += ["--copies", str(cfg.get("copies", 1)), "--seed", str(seed)]
    for split, n in (cfg.get("split_copies") or {}).items():
        argv += ["--split-copies", f"{split}={n}"]
    return argv


def read_annotators(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _first_assignments(ann: Path, annotators: Path, cfg: dict) -> None:
    """``eval_assignment/`` and ``train_assignment/`` for the annotators listed now, each written once."""
    seed = cfg.get("seed", 0)
    for kind, _, name in LEGACY:
        source = ann / ("eval_proposal.json" if kind == EVAL else "train_annotation_sample.json")
        batch_cfg = cfg.get(kind, {})
        inputs = json.dumps(
            {"source": files_fingerprint([source]), "annotators": files_fingerprint([annotators]),
             "settings": {k: v for k, v in batch_cfg.items() if k in ("copies", "split_copies")}, "seed": seed},
            sort_keys=True,
        )
        argv = _assign_argv(source, annotators, batch_cfg, seed, ann / name)
        write_once(ann / name, inputs, lambda argv=argv, name=name: check_rc(assign.main(argv), f"assign {name}"))


def train_bundle_settings(split_config_path: str) -> dict:
    cfg = load_split_config(split_config_path).get("assignments", {})
    train = cfg.get("train", {})
    return {
        "size": train.get("bundle_size", 200),
        "calibration": train.get("calibration_episodes", 20),
        "seed": cfg.get("seed", 0),
    }


def returned_uids(annotation_config_path: str, data_dir: str) -> set[str]:
    """uids with a collected annotation in the train store."""
    cfg = load_config(Path(annotation_config_path), Path(data_dir))
    stores = [s.store for s in cfg.stores if s.kind == TRAIN]
    return {uid for s in stores for uid in current_annotations(read_store(s))}


@stage(always_run=True)
def annotation_roster(split_config_path: str, annotation_config_path: str, data_dir: str) -> str:
    """Fingerprint of ``annotators.txt`` and the roster."""
    ann = annotation_dir(Path(data_dir))
    cfg = load_split_config(split_config_path).get("assignments", {})
    roster = load_config(Path(annotation_config_path), Path(data_dir)).bundles_dir / ROSTER_NAME
    return files_fingerprint([ann / cfg.get("annotators_file", "annotators.txt"), roster])


@stage
def annotation_bundles(
    chosen_proposal: str,
    train_sample: str,
    annotation_roster: str,
    split_config: str,
    split_config_path: str,
    annotation_config_path: str,
    data_dir: str,
    index_path: str,
) -> str:
    """The bundles and roster in the annotation config's ``bundles_dir``; returns their fingerprint.

    Nothing is assigned until a proposal is chosen and ``assignments.annotators_file`` exists.
    Existing ``eval_assignment/`` and ``train_assignment/`` folders are adopted as the first bundles.
    """
    if chosen_proposal == NO_PROPOSAL:
        return NO_ASSIGNMENTS
    ann = annotation_dir(Path(data_dir))
    cfg = load_split_config(split_config_path).get("assignments", {})
    annotators = ann / cfg.get("annotators_file", "annotators.txt")
    bundles_dir = load_config(Path(annotation_config_path), Path(data_dir)).bundles_dir
    registry = Registry(bundles_dir)
    if not registry.exists:
        if not all((ann / name).is_dir() for _, _, name in LEGACY):
            if not annotators.is_file():
                print(f"waiting for annotators: write one name per line to {annotators}", file=sys.stderr)
                return NO_ASSIGNMENTS
            _first_assignments(ann, annotators, cfg)
        registry = adopt(bundles_dir, [(kind, slot, ann / name) for kind, slot, name in LEGACY])
        print(f"bundles: {len(registry.bundles)} bundle(s) from {', '.join(n for *_, n in LEGACY)}", file=sys.stderr)

    newcomers = [a for a in read_annotators(annotators) if a not in registry.annotators()]
    if newcomers:
        settings = train_bundle_settings(split_config_path)
        pool = TrainPool(Path(index_path), ann / "eval_proposal.json")
        returned = returned_uids(annotation_config_path, data_dir)
        for name in newcomers:
            try:
                h = add_train_bundle(registry, pool, name, returned=returned, note="registered", **settings)
            except ValueError as exc:
                print(f"bundles: skipped {name!r} from {annotators.name}: {exc}", file=sys.stderr)
                continue
            print(f"bundles: {h.bundle} ({len(registry.bundles[h.bundle]['episodes'])} train episodes) -> {name}, {h.slot}", file=sys.stderr)
    return files_fingerprint(registry.files())


@stage(always_run=True)
def annotation_state(annotation_config_path: str, data_dir: str) -> str:
    """Fingerprint of the annotation config and every annotation store."""
    cfg = load_config(Path(annotation_config_path), Path(data_dir))
    return files_fingerprint([Path(annotation_config_path), *(s.store for s in cfg.stores)])


@stage
def annotation_columns(
    annotation_bundles: str, annotation_state: str, annotation_config_path: str, data_dir: str, index_path: str
) -> str:
    """The ``annotation_*`` columns; returns their fingerprint.

    Without bundles the index is left unchanged.
    """
    if annotation_bundles == NO_ASSIGNMENTS:
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
