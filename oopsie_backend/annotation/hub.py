"""Distribute annotation bundles through a Hugging Face dataset repo and pull returns from its pull requests.

The batch repo holds ``<folder>/<slot>/`` for every current holding of the roster
(``annotation.roster``). Annotators return their annotated h5 files, at the same
paths, as pull requests to a separate returns repo (``oopsie-data annotation upload``).
A pull request is accepted when it only changes ``<folder>/<slot>/episodes/NNNN.h5``
files of a single annotator folder, in slots the roster gives or gave that folder;
accepted pull requests are downloaded to ``<returns_dir>/pr_NNNN/``. A pull request
is downloaded at its head commit, which ``.pr_commit`` records, and again only when the head changes. The files an open pull request
changes are found by comparing its tree with the main branch; a merged one's files are
on the main branch already, so they are read from the pull request's diff.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

from oopsie_backend.annotation.roster import KINDS, Registry
from oopsie_backend.common.repos import annotation_dir

COMMIT_MARKER = ".pr_commit"
PULLED_STATUSES = ("open", "merged", "draft")


@dataclass
class Store:
    kind: str
    store: Path
    release_dir: Path


@dataclass
class PipelineConfig:
    repo_id: str
    returns_repo_id: str
    data_dir: Path
    bundles_dir: Path
    stage_dir: Path
    returns_dir: Path
    index: Path | None = None
    stores: list[Store] = field(default_factory=list)

    def registry(self) -> Registry:
        return Registry(self.bundles_dir)


def load_config(path: Path, data_dir: Path | None = None, index: Path | None = None) -> PipelineConfig:
    """The config at ``path``; relative paths resolve against ``annotation_dir(data_dir)``.

    ``data_dir`` and ``index`` override the config's values.
    """
    raw = yaml.safe_load(path.read_text())
    data_dir = data_dir or Path(raw["data_dir"])
    root = annotation_dir(data_dir)
    index = index or (Path(raw["index"]) if raw.get("index") else None)
    cfg = PipelineConfig(
        repo_id=raw["repo_id"],
        returns_repo_id=raw["returns_repo_id"],
        data_dir=data_dir,
        bundles_dir=root / raw["bundles_dir"],
        stage_dir=root / raw["stage_dir"],
        returns_dir=root / raw["returns_dir"],
        index=index,
        stores=[Store(kind, root / s["store"], root / s["release_dir"]) for kind, s in raw["stores"].items()],
    )
    unknown = [s.kind for s in cfg.stores if s.kind not in KINDS]
    if unknown:
        raise ValueError(f"{path}: unknown store kinds {unknown}; expected {KINDS}")
    return cfg


def annotator_slots(registry: Registry) -> dict[str, set[str]]:
    """Repo folder -> every slot the roster gives or gave it."""
    out: dict[str, set[str]] = {}
    for h in registry.holdings:
        out.setdefault(h.folder, set()).add(h.slot)
    return out


def file_ids(api: HfApi, repo: str, revision: str | None) -> dict[str, str]:
    return {
        f.path: f.blob_id
        for f in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
        if isinstance(f, RepoFile)
    }


def changed_paths(api: HfApi, repo: str, revision: str, base: dict[str, str]) -> set[str]:
    """Files at ``revision`` that are new or differ from ``base`` (path -> blob id)."""
    return {path for path, blob in file_ids(api, repo, revision).items() if base.get(path) != blob}


DIFF_PATH_RE = re.compile(r"diff --git a/\S+ b/(\S+)")


def merged_changes(api: HfApi, repo: str, num: int) -> tuple[set[str], str | None]:
    """Files a merged pull request added or changed, from its diff, and its merge commit."""
    details = api.get_discussion_details(repo, num, repo_type="dataset")
    blocks = re.split(r"(?=^diff --git )", details.diff or "", flags=re.MULTILINE)
    paths = {m.group(1) for b in blocks if (m := DIFF_PATH_RE.match(b)) and "\ndeleted file mode" not in b}
    return paths, details.merge_commit_oid


def check_pull_request(paths: set[str], slots: dict[str, set[str]]) -> tuple[str | None, str | None]:
    """``(folder, None)`` for an acceptable change set, else ``(None, reason)``."""
    if not paths:
        return None, "no changed files"
    tops = {p.split("/", 1)[0] for p in paths}
    if len(tops) != 1:
        return None, f"touches several folders: {sorted(tops)}"
    folder = tops.pop()
    if folder not in slots:
        return None, f"unknown folder {folder!r}"
    allowed = re.compile(rf"^{re.escape(folder)}/({'|'.join(map(re.escape, sorted(slots[folder])))})/episodes/\d+\.h5$")
    bad = sorted(p for p in paths if not allowed.match(p))
    if bad:
        return None, f"changes files other than episode h5s: {bad[:5]}"
    return folder, None


def pull_returns(
    api: HfApi, cfg: PipelineConfig, slots: dict[str, set[str]], close: bool = False
) -> list[dict]:
    """Download every acceptable pull request of the returns repo that changed since the last pull.

    Returns one row per pull request.
    """
    repo = cfg.returns_repo_id
    heads = {
        ref.ref: ref.target_commit
        for ref in api.list_repo_refs(repo, repo_type="dataset", include_pull_requests=True).pull_requests
    }
    base = file_ids(api, repo, None)
    rows = []
    for d in api.get_repo_discussions(repo, repo_type="dataset", discussion_type="pull_request"):
        if d.status not in PULLED_STATUSES:
            continue
        row = {"num": d.num, "author": d.author, "status": d.status, "title": d.title}
        revision = f"refs/pr/{d.num}"
        head = heads.get(revision)
        if d.status == "merged":
            paths, merge_commit = merged_changes(api, repo, d.num)
            if not head and merge_commit:
                revision, head = merge_commit, merge_commit
        else:
            paths = changed_paths(api, repo, revision, base)
        folder, reason = check_pull_request(paths, slots)
        row |= {"folder": folder, "files": len(paths)}
        if reason:
            rows.append(row | {"result": f"rejected: {reason}"})
            continue
        if not head:
            try:
                head = api.repo_info(repo, repo_type="dataset", revision=revision).sha
            except Exception as exc:
                rows.append(row | {"result": f"skipped: head commit of {revision} not found ({exc})"})
                continue
        dest = cfg.returns_dir / f"pr_{d.num:04d}"
        marker = dest / COMMIT_MARKER
        if marker.is_file() and marker.read_text().strip() == head:
            result = "unchanged"
        else:
            api.snapshot_download(
                repo, repo_type="dataset", revision=head, local_dir=dest, allow_patterns=sorted(paths)
            )
            marker.write_text(head)
            result = "downloaded"
        if close and d.status == "open":
            api.change_discussion_status(
                repo, d.num, "closed", repo_type="dataset", comment="Annotations collected, thank you."
            )
            result += ", closed"
        rows.append(row | {"result": result})
    (cfg.returns_dir / "pull_requests.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.returns_dir / "pull_requests.json").write_text(json.dumps(rows, indent=2))
    return rows


def returned_slot_dirs(cfg: PipelineConfig) -> list[Path]:
    """Completely downloaded ``pr_NNNN/<folder>/<slot>`` directories, oldest pull request first."""
    return sorted(
        p for p in cfg.returns_dir.glob("pr_*/*/*") if p.is_dir() and (p.parents[1] / COMMIT_MARKER).is_file()
    )
