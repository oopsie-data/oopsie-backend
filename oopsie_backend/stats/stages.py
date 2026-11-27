"""Pipeline stages of the public and internal stats and of the dataset analysis."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.annotation.hub import load_config as load_annotation_config
from oopsie_backend.annotation.stages import NO_ASSIGNMENTS
from oopsie_backend.common.stats_config import DEFAULT_CONFIG_PATH
from oopsie_backend.pipeline.registry import check_rc, files_fingerprint, stage
from oopsie_backend.stats import compute_local
from oopsie_backend.stats.analysis import common as analysis_common
from oopsie_backend.stats.analysis import run as analysis_run


@stage(always_run=True)
def stats_config() -> str:
    """Fingerprint of ``configs/stats_config.yaml``."""
    return files_fingerprint([DEFAULT_CONFIG_PATH])


@stage
def stats(
    critical_episodes: str, stats_config: str, data_dir: str, index_path: str, stats_docs_output: str, reports_dir: str
) -> str:
    """Writes the public and internal stats; returns the fingerprint of the docs ``stats.json``."""
    reports = Path(reports_dir)
    argv = [
        "--data-dir", data_dir,
        "--index", index_path,
        "--docs-output", stats_docs_output,
        "--public-json", str(reports / "public_stats.json"),
        "--public-html", str(reports / "public_stats.html"),
        "--internal-output", str(reports / "internal_stats.json"),
    ]
    check_rc(compute_local.main(argv), "stats")
    return files_fingerprint([Path(stats_docs_output)])


@stage(always_run=True, manual=True)
def analysis_config() -> str:
    """Fingerprint of ``configs/analysis.yaml``."""
    return files_fingerprint([analysis_common.ANALYSIS_CONFIG])


@stage(manual=True)
def dataset_analysis(
    critical_episodes: str,
    annotation_flags: str,
    unified_camera_names: str,
    unified_embodiments: str,
    unified_policies: str,
    analysis_config: str,
    index_path: str,
    reports_dir: str,
) -> str:
    """Writes the index-based dataset analysis, ``<reports>/analysis/index.json``; returns its fingerprint.

    Needs the ``analysis`` extra.
    """
    out = Path(reports_dir) / "analysis" / "index.json"
    check_rc(analysis_run.main(["index", "--index", index_path, "--out", str(out)]), "dataset analysis")
    return files_fingerprint([out])


@stage(manual=True)
def annotation_analysis(
    annotation_flags: str,
    critical_episodes: str,
    unified_camera_names: str,
    unified_embodiments: str,
    unified_policies: str,
    analysis_config: str,
    annotation_config_path: str,
    data_dir: str,
    index_path: str,
    reports_dir: str,
) -> str:
    """Writes the annotation-based analysis per batch and over all batches; returns the fingerprint of the reports.

    ``<reports>/analysis/annotations_<batch>.json`` and ``annotations.json``. Needs the ``analysis`` extra.
    """
    if annotation_flags == NO_ASSIGNMENTS:
        return NO_ASSIGNMENTS
    cfg = load_annotation_config(Path(annotation_config_path), Path(data_dir))
    out_dir = Path(reports_dir) / "analysis"
    runs = {f"annotations_{s.kind}.json": ["--batch", s.kind] for s in cfg.stores}
    runs["annotations.json"] = []
    for name, batch in runs.items():
        argv = ["annotations", "--index", index_path, *batch, "--out", str(out_dir / name)]
        check_rc(analysis_run.main(argv), f"annotation analysis {name}")
    return files_fingerprint([out_dir / name for name in runs])
