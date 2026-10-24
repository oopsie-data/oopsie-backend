"""Dataset repos of the HF org, their kinds, and where episodes live on disk.

``configs/repositories.yaml`` lists the repos that are not Oopsie contributions.
Each repo has a kind: ``contributor`` (every repo not listed), ``external``
(imported datasets, never public), ``archive`` (frozen snapshots) or ``infra``
(site assets and the annotation repo). Only contributor and external repos hold
episodes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from oopsie_backend.common.stats_config import CONFIG

REPOSITORIES_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "repositories.yaml"
_REPOS = yaml.safe_load(REPOSITORIES_CONFIG.read_text()) or {}

EXTERNAL_DATASETS: frozenset[str] = frozenset(_REPOS.get("external") or [])
ARCHIVE_DATASETS: frozenset[str] = frozenset(_REPOS.get("archive") or [])
INFRA_DATASETS: frozenset[str] = frozenset(_REPOS.get("infra") or [])

# Names used by the stats code.
STATS_EXCLUDED_DATASETS = ARCHIVE_DATASETS
PUBLIC_EXCLUDED_DATASETS = EXTERNAL_DATASETS
PER_LAB_EXCLUDED_DATASETS = EXTERNAL_DATASETS
NON_LAB_INFRA_DATASETS = INFRA_DATASETS
# Placeholder / unset lab_ids omitted from the public website stats.
PUBLIC_EXCLUDED_LABS: frozenset[str] = CONFIG["public_excluded_labs"]

HF_ORG = "OopsieData-Submissions"
DEFAULT_DATA_DIR = Path("/corral/utexas/IRI26024/oopsiedata")

CONTRIBUTOR = "contributor"
EXTERNAL = "external"
ARCHIVE = "archive"
INFRA = "infra"
EPISODE_KINDS = (CONTRIBUTOR, EXTERNAL)

# Folders inside a repo that never hold submissions: the HF download cache, and
# qa_out, where QA review bundles are unpacked.
IGNORED_DIR_NAMES = frozenset({".cache", "qa_out"})


def repo_kind(dataset_repo: str) -> str:
    if dataset_repo in ARCHIVE_DATASETS:
        return ARCHIVE
    if dataset_repo in INFRA_DATASETS:
        return INFRA
    if dataset_repo in EXTERNAL_DATASETS:
        return EXTERNAL
    return CONTRIBUTOR


def org_dir(data_dir: Path) -> Path:
    """The ``HF_ORG`` folder of a mirror; ``data_dir`` may be the mirror root or that folder."""
    return data_dir if data_dir.name == HF_ORG else data_dir / HF_ORG


def annotation_dir(data_dir: Path) -> Path:
    """``<mirror root>/annotation``: assignments, bundles, returns and stores, next to the org folder."""
    root = data_dir.parent if data_dir.name == HF_ORG else data_dir
    return root / "annotation"


DEFAULT_ANNOTATION_DIR = annotation_dir(DEFAULT_DATA_DIR)


def episode_path(data_dir: Path, dataset_repo: str, repo_path: str) -> Path:
    return org_dir(data_dir) / dataset_repo.split("/", 1)[1] / repo_path


def canonical_dataset_repo(dataset: str) -> str:
    """Normalize any dataset label to ``namespace/repo`` for comparisons."""
    if dataset.startswith("local:"):
        path_parts = Path(dataset[6:]).parts
        for i, part in enumerate(path_parts):
            if part == HF_ORG and i + 1 < len(path_parts):
                return f"{HF_ORG}/{path_parts[i + 1]}"
        return dataset

    parts = dataset.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == HF_ORG:
        return f"{parts[0]}/{parts[1]}"
    if parts and parts[0] != HF_ORG:
        return f"{HF_ORG}/{parts[0]}"
    return dataset


def resolve_local_episode_path(
    data_dir: Path, rel: Path
) -> tuple[str, str] | None:
    """Map a local ``.h5`` path to ``(dataset_repo, repo_relative_path)``."""
    parts = rel.parts
    if data_dir.name == HF_ORG:
        if len(parts) < 2:
            return None
        return f"{HF_ORG}/{parts[0]}", "/".join(parts[1:])
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def discover_episodes(data_dir: Path) -> list[tuple[str, str, Path]]:
    """``(dataset_repo, repo_path, absolute_path)`` for every .h5 under the org folder."""
    root = org_dir(data_dir.resolve())
    episodes: list[tuple[str, str, Path]] = []
    for path in sorted(root.rglob("*.h5")):
        rel = path.relative_to(root)
        if IGNORED_DIR_NAMES.intersection(rel.parts):
            continue
        resolved = resolve_local_episode_path(root, rel)
        if resolved is None:
            continue
        dataset_repo, repo_path = resolved
        episodes.append((dataset_repo, repo_path, path))
    return episodes


def discover_all_dataset_dirs(data_dir: Path) -> list[str]:
    """Every dataset repo id on disk, including ones without .h5 files."""
    root = org_dir(data_dir.resolve())
    if not root.is_dir():
        return []
    return sorted(f"{HF_ORG}/{p.name}" for p in root.iterdir() if p.is_dir())
