"""Dataset repos of the HF org, their kinds, and where episodes live on disk.

``configs/repositories.yaml`` lists the archive and infra repos and
``configs/imports.yaml`` the imported datasets. Each dataset has a kind:
``contributor`` (every org repo not listed), ``external`` (imported datasets,
never public), ``archive`` (frozen snapshots) or ``infra`` (site assets and the
annotation repos). Only contributor and external datasets hold episodes.

Imported datasets are either repos of the org or converted locally into
``<data root>/imports/<name>``; the latter have the dataset id ``imports/<name>``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from oopsie_backend.common.h5 import find_h5
from oopsie_backend.common.stats_config import CONFIG

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
REPOSITORIES_CONFIG = CONFIG_DIR / "repositories.yaml"
IMPORTS_CONFIG = CONFIG_DIR / "imports.yaml"
LAB_IDS_CONFIG = CONFIG_DIR / "lab_ids.yaml"
_REPOS = yaml.safe_load(REPOSITORIES_CONFIG.read_text()) or {}
IMPORTS_PREFIX = "imports"


def load_imports(path: Path = IMPORTS_CONFIG) -> dict[str, dict]:
    """Entries of ``configs/imports.yaml`` by name; each has ``hub`` or ``converter`` and ``source``."""
    datasets = (yaml.safe_load(path.read_text()) or {}).get("datasets") or {}
    for name, entry in datasets.items():
        if bool(entry.get("hub")) == bool(entry.get("converter")):
            raise ValueError(f"{path}: {name} needs exactly one of hub or converter")
        if entry.get("converter") and not entry.get("source"):
            raise ValueError(f"{path}: {name} needs a source")
    return datasets


def lab_key(lab_id: str) -> str:
    """Spelling-insensitive form of a lab_id: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", lab_id.casefold())


def load_lab_aliases(path: Path = LAB_IDS_CONFIG) -> dict[str, str]:
    """``lab_key`` of every canonical and raw lab_id -> canonical lab_id, from ``configs/lab_ids.yaml``."""
    config = yaml.safe_load(path.read_text()) or {}
    aliases = {}
    for canonical, raw_names in config.items():
        for raw in [canonical, *(raw_names or [])]:
            key = lab_key(raw)
            if aliases.get(key, canonical) != canonical:
                raise ValueError(f"{path}: {raw} is listed under {aliases[key]} and {canonical}")
            aliases[key] = canonical
    return aliases


def canonical_lab_id(lab_id: str | None, aliases: dict[str, str]) -> str | None:
    """Canonical lab_id for ``lab_id`` in any spelling ``aliases`` covers; other lab_ids are returned unchanged."""
    return aliases.get(lab_key(lab_id), lab_id) if lab_id else lab_id


def import_dataset_id(name: str, entry: dict) -> str:
    return entry["hub"] if entry.get("hub") else f"{IMPORTS_PREFIX}/{name}"


EXTERNAL_DATASETS: frozenset[str] = frozenset(import_dataset_id(n, e) for n, e in load_imports().items())
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


def is_local_import(dataset_repo: str) -> bool:
    return dataset_repo.startswith(f"{IMPORTS_PREFIX}/")


def repo_kind(dataset_repo: str) -> str:
    if is_local_import(dataset_repo):
        return EXTERNAL
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


def data_root(data_dir: Path) -> Path:
    """The mirror root; ``data_dir`` may be the root or its ``HF_ORG`` folder."""
    return data_dir.parent if data_dir.name == HF_ORG else data_dir


def annotation_dir(data_dir: Path) -> Path:
    """``<mirror root>/annotation``: assignments, bundles, returns and stores, next to the org folder."""
    return data_root(data_dir) / "annotation"


def imports_dir(data_dir: Path) -> Path:
    """``<mirror root>/imports``: locally converted datasets, next to the org folder."""
    return data_root(data_dir) / IMPORTS_PREFIX


DEFAULT_ANNOTATION_DIR = annotation_dir(DEFAULT_DATA_DIR)


def episode_path(data_dir: Path, dataset_repo: str, repo_path: str) -> Path:
    if is_local_import(dataset_repo):
        return data_root(data_dir) / dataset_repo / repo_path
    return org_dir(data_dir) / dataset_repo.split("/", 1)[1] / repo_path


def canonical_dataset_repo(dataset: str) -> str:
    """Normalize any dataset label to ``namespace/repo`` for comparisons."""
    if is_local_import(dataset):
        return dataset
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
    """Map a local HDF5 path to ``(dataset_repo, repo_relative_path)``."""
    parts = rel.parts
    if data_dir.name == HF_ORG:
        if len(parts) < 2:
            return None
        return f"{HF_ORG}/{parts[0]}", "/".join(parts[1:])
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def discover_episodes(data_dir: Path) -> list[tuple[str, str, Path]]:
    """``(dataset_repo, repo_path, absolute_path)`` for every HDF5 file under the org folder."""
    root = org_dir(data_dir.resolve())
    episodes: list[tuple[str, str, Path]] = []
    for path in find_h5(root):
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
    """Every dataset repo id on disk, including ones without HDF5 files."""
    root = org_dir(data_dir.resolve())
    if not root.is_dir():
        return []
    return sorted(f"{HF_ORG}/{p.name}" for p in root.iterdir() if p.is_dir())
