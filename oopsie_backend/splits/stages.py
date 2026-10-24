"""Pipeline stages of the eval split: proposals, the chosen proposal, the train sample and the index columns.

Settings come from ``configs/eval_split.yaml``; outputs go to ``<data_dir>/annotation/``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from oopsie_backend.common.repos import annotation_dir
from oopsie_backend.index.store import columns_fingerprint
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage, write_once
from oopsie_backend.splits import propose, record, sample_train
from oopsie_backend.splits.columns import EVAL_COLUMNS

NO_PROPOSAL = "no proposal"


def load_split_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


@stage(always_run=True)
def split_config(split_config_path: str) -> str:
    """Fingerprint of ``configs/eval_split.yaml``."""
    return files_fingerprint([Path(split_config_path)])


@stage
def eval_proposals(critical_episodes: str, split_config: str, split_config_path: str, data_dir: str, index_path: str) -> str:
    """Candidate proposals in ``annotation/proposals/``, from episodes that are neither duplicate nor critical."""
    out = annotation_dir(Path(data_dir)) / "proposals"
    argv = ["--index", index_path, "--out-dir", str(out), *load_split_config(split_config_path).get("propose_args", [])]
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
def eval_split(chosen_proposal: str, data_dir: str, index_path: str) -> str:
    """``eval_split`` and ``eval_proposal`` from the chosen proposal; returns their fingerprint.

    Without a chosen proposal the index is left unchanged.
    """
    if chosen_proposal == NO_PROPOSAL:
        return NO_PROPOSAL
    proposal = annotation_dir(Path(data_dir)) / "eval_proposal.json"
    check_rc(record.main(["--proposal", str(proposal), "--index", index_path]), "record eval split")
    return columns_fingerprint(Path(index_path), EVAL_COLUMNS)
