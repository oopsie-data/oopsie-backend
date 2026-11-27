"""Pipeline stages of the eval split: proposals, the chosen proposal, the train sample and the index columns.

Settings come from ``configs/eval_split.yaml``; outputs go to ``<data_dir>/annotation/``.
``chosen_proposal`` is a proposal id from ``annotation/proposals/`` or, ending in ``.json``,
a proposal file relative to the backend repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

from oopsie_backend.common.repos import CONFIG_DIR, annotation_dir
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage, write_once
from oopsie_backend.splits import propose, record, sample_train
from oopsie_backend.splits.columns import EVAL_COLUMNS
from oopsie_backend.splits.eval_split import DEFAULT_ROBOT_ALIASES

NO_PROPOSAL = "no proposal"


def load_split_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def proposal_source(chosen: str | None) -> Path | None:
    """The repo proposal file ``chosen`` names, or None for a proposal id."""
    if not chosen or not chosen.endswith(".json"):
        return None
    path = Path(chosen)
    return path if path.is_absolute() else CONFIG_DIR.parent / path


def freeze_proposal(source: Path, out: Path) -> str:
    """Copy ``source`` to ``out`` once; an existing ``out`` must hold the same proposal."""
    if not source.is_file():
        raise FileNotFoundError(f"chosen_proposal names {source}, which does not exist")
    record = out.with_name(out.name + ".inputs")
    inputs = json.dumps({"proposal_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    if out.exists():
        if json.loads(out.read_text()) != json.loads(source.read_text()):
            raise RuntimeError(f"{out} holds a different proposal than {source}")
    else:
        shutil.copyfile(source, out)
    if not record.is_file() or record.read_text() != inputs:
        record.write_text(inputs)
    return files_fingerprint([out])


@stage(always_run=True)
def split_config(split_config_path: str) -> str:
    """Fingerprint of ``configs/eval_split.yaml`` and the repo proposal file it names."""
    source = proposal_source(load_split_config(split_config_path).get("chosen_proposal"))
    return files_fingerprint([Path(split_config_path), *([source] if source and source.is_file() else [])])


def robot_aliases_path(propose_args: list[str]) -> Path:
    """The ``--robot-aliases`` file ``propose_args`` passes, else the default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot-aliases", type=Path, default=DEFAULT_ROBOT_ALIASES)
    return parser.parse_known_args(propose_args)[0].robot_aliases


@stage(always_run=True)
def robot_aliases_config(split_config_path: str) -> str:
    """Fingerprint of the robot aliases the proposal search reads (``configs/robot_aliases.yaml``)."""
    return files_fingerprint([robot_aliases_path(load_split_config(split_config_path).get("propose_args", []))])


@stage
def eval_proposals(
    critical_episodes: str, split_config: str, robot_aliases_config: str, split_config_path: str, data_dir: str, index_path: str
) -> str:
    """Candidate proposals in ``annotation/proposals/``, from episodes that are neither duplicate nor critical.

    Once a proposal is chosen (``annotation/eval_proposal.json`` exists or ``chosen_proposal`` names a
    repo file), the proposals are kept as they are.
    """
    ann = annotation_dir(Path(data_dir))
    out = ann / "proposals"
    cfg = load_split_config(split_config_path)
    if (ann / "eval_proposal.json").is_file() or proposal_source(cfg.get("chosen_proposal")):
        return files_fingerprint(sorted(out.glob("*.json")))
    argv = ["--index", index_path, "--out-dir", str(out), *cfg.get("propose_args", [])]
    check_rc(propose.main(argv), "propose eval split")
    return files_fingerprint(sorted(out.glob("*.json")))


@stage
def chosen_proposal(eval_proposals: str, split_config: str, split_config_path: str, data_dir: str) -> str:
    """``annotation/eval_proposal.json``: the chosen proposal, copied once and then kept."""
    ann = annotation_dir(Path(data_dir))
    out = ann / "eval_proposal.json"
    chosen = load_split_config(split_config_path).get("chosen_proposal")
    if not chosen and not out.exists():
        print(
            f"waiting for a choice: review {ann / 'proposals' / 'REPORT.md'} and set chosen_proposal in {split_config_path}",
            file=sys.stderr,
        )
        return NO_PROPOSAL
    source = proposal_source(chosen)
    if source:
        return freeze_proposal(source, out)
    return write_once(out, json.dumps({"chosen_proposal": chosen}), lambda: shutil.copyfile(ann / "proposals" / f"{chosen}.json", out))


@stage
def train_sample(chosen_proposal: str, split_config: str, split_config_path: str, data_dir: str, index_path: str) -> str:
    """``annotation/train_annotation_sample.json``, sampled once from the chosen proposal's train set."""
    if chosen_proposal == NO_PROPOSAL:
        return NO_PROPOSAL
    ann = annotation_dir(Path(data_dir))
    cfg = load_split_config(split_config_path).get("train_sample", {})
    out = ann / "train_annotation_sample.json"
    argv = ["--index", index_path, "--proposal", str(ann / "eval_proposal.json"), "--out", str(out)]
    argv += ["--num-episodes", str(cfg.get("num_episodes", 1000))]
    inputs = json.dumps({"proposal": chosen_proposal, "train_sample": cfg}, sort_keys=True)
    return write_once(out, inputs, lambda: check_rc(sample_train.main(argv), "train sample"))


@stage
def eval_split(chosen_proposal: str, episode_index: str, data_dir: str, index_path: str) -> str:
    """``eval_split`` and ``eval_proposal`` from the chosen proposal and the index; returns their fingerprint.

    Without a chosen proposal the index is left unchanged.
    """
    if chosen_proposal == NO_PROPOSAL:
        return NO_PROPOSAL
    proposal = annotation_dir(Path(data_dir)) / "eval_proposal.json"
    check_rc(record.main(["--proposal", str(proposal), "--index", index_path]), "record eval split")
    return columns_fingerprint(Path(index_path), EVAL_COLUMNS)
