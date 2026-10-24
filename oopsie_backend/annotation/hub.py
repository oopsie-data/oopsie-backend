"""Distribute annotation batches through a Hugging Face dataset repo and pull returns from its pull requests.

The repo holds ``<folder>/<batch>/`` per annotator (see ``stage_annotation_repo``).
Annotators return their annotated h5 files as pull requests. A pull request is
accepted when it only changes ``<folder>/<batch>/episodes/NNNN.h5`` files of a
single annotator folder; accepted pull requests are downloaded to
``<returns_dir>/pr_NNNN/`` together with that folder's manifests. A pull
request is downloaded again only when its head commit changes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from oopsie_backend.annotation.bundles import MANIFEST_NAME
from oopsie_backend.common.repos import annotation_dir

COMMIT_MARKER = ".pr_commit"
PULLED_STATUSES = ("open", "merged")
_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


@dataclass
class Batch:
    name: str
    assignment_dir: Path
    store: Path
    release_dir: Path


@dataclass
class PipelineConfig:
    repo_id: str
    data_dir: Path
    stage_dir: Path
    returns_dir: Path
    index: Path | None = None
    batches: list[Batch] = field(default_factory=list)


def load_config(path: Path, data_dir: Path | None = None, index: Path | None = None) -> PipelineConfig:
    """The config at ``path``; relative paths resolve against ``annotation_dir(data_dir)``.

    ``data_dir`` and ``index`` override the config's values.
    """
    raw = yaml.safe_load(path.read_text())
    data_dir = data_dir or Path(raw["data_dir"])
    root = annotation_dir(data_dir)
    index = index or (Path(raw["index"]) if raw.get("index") else None)
    return PipelineConfig(
        repo_id=raw["repo_id"],
        data_dir=data_dir,
        stage_dir=root / raw["stage_dir"],
        returns_dir=root / raw["returns_dir"],
        index=index,
        batches=[
            Batch(
                name=name,
                assignment_dir=root / b["assignment_dir"],
                store=root / b["store"],
                release_dir=root / b["release_dir"],
            )
            for name, b in raw["batches"].items()
        ],
    )


def folder_name(annotator: str) -> str:
    """Annotator name as lowercase ASCII with ``_`` separators."""
    ascii_name = unicodedata.normalize("NFKD", annotator).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def annotator_folders(cfg: PipelineConfig) -> dict[str, str]:
    """Annotator name -> repo folder, over all batches."""
    out = {}
    for b in cfg.batches:
        for path in sorted(b.assignment_dir.glob("*.json")):
            if path.name != "assignment_summary.json":
                name = json.loads(path.read_text())["annotator"]
                out[name] = folder_name(name)
    return out


def changed_paths(diff: str) -> set[str]:
    """Paths touched by a git diff."""
    return {p for m in _DIFF_HEADER.finditer(diff) for p in m.groups()}


def check_pull_request(paths: set[str], folders: set[str], batches: list[str]) -> tuple[str | None, str | None]:
    """``(folder, None)`` for an acceptable change set, else ``(None, reason)``."""
    if not paths:
        return None, "no changed files"
    tops = {p.split("/", 1)[0] for p in paths}
    if len(tops) != 1:
        return None, f"touches several folders: {sorted(tops)}"
    folder = tops.pop()
    if folder not in folders:
        return None, f"unknown folder {folder!r}"
    allowed = re.compile(rf"^{re.escape(folder)}/({'|'.join(map(re.escape, batches))})/episodes/\d+\.h5$")
    bad = sorted(p for p in paths if not allowed.match(p))
    if bad:
        return None, f"changes files other than episode h5s: {bad[:5]}"
    return folder, None


def pull_returns(
    api: HfApi, cfg: PipelineConfig, folders: set[str], close: bool = False
) -> list[dict]:
    """Download every acceptable pull request that changed since the last pull; return one row per pull request."""
    batches = [b.name for b in cfg.batches]
    heads = {
        ref.ref: ref.target_commit
        for ref in api.list_repo_refs(cfg.repo_id, repo_type="dataset", include_pull_requests=True).pull_requests
    }
    rows = []
    for d in api.get_repo_discussions(cfg.repo_id, repo_type="dataset", discussion_type="pull_request"):
        if d.status not in PULLED_STATUSES:
            continue
        row = {"num": d.num, "author": d.author, "status": d.status, "title": d.title}
        details = api.get_discussion_details(cfg.repo_id, d.num, repo_type="dataset")
        paths = changed_paths(details.diff or "")
        folder, reason = check_pull_request(paths, folders, batches)
        row |= {"folder": folder, "files": len(paths)}
        if reason:
            rows.append(row | {"result": f"rejected: {reason}"})
            continue
        dest = cfg.returns_dir / f"pr_{d.num:04d}"
        head = heads.get(f"refs/pr/{d.num}")
        marker = dest / COMMIT_MARKER
        if head and marker.is_file() and marker.read_text().strip() == head:
            result = "unchanged"
        else:
            touched = sorted({p.split("/")[1] for p in paths})
            api.snapshot_download(
                cfg.repo_id,
                repo_type="dataset",
                revision=f"refs/pr/{d.num}",
                local_dir=dest,
                allow_patterns=sorted(paths) + [f"{folder}/{b}/{MANIFEST_NAME}" for b in touched],
            )
            if head:
                marker.write_text(head)
            result = "downloaded"
        if close and d.status == "open":
            api.change_discussion_status(
                cfg.repo_id, d.num, "closed", repo_type="dataset", comment="Annotations collected, thank you."
            )
            result += ", closed"
        rows.append(row | {"result": result})
    (cfg.returns_dir / "pull_requests.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.returns_dir / "pull_requests.json").write_text(json.dumps(rows, indent=2))
    return rows


def returned_batch_dirs(cfg: PipelineConfig, batch: str) -> list[Path]:
    """Downloaded ``pr_NNNN/<folder>/<batch>`` directories, oldest pull request first."""
    return sorted(p for p in cfg.returns_dir.glob(f"pr_*/*/{batch}") if p.is_dir())
