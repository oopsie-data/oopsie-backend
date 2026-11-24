"""Compute the dataset analysis of plans/stats.md, in three groups.

Usage:
    uv run oopsie stats analysis index --index episode_index.parquet --out reports/analysis/index.json
    uv run oopsie stats analysis annotations --index episode_index.parquet [--batch NAME ...] --out reports/analysis/annotations.json
    uv run oopsie stats analysis features --index episode_index.parquet --data-dir <mirror> --out <features.parquet> [--segments]
    uv run oopsie stats analysis data --index episode_index.parquet --features <features.parquet> --out reports/analysis/data.json
    uv run oopsie stats analysis render reports/analysis/*.json

``index`` uses the episode index alone. ``annotations`` reads the annotators' returns with
failure segments from the index's ``annotation_*`` columns (``from_annotations``), of every
batch or of the ``--batch`` names. ``features`` reads the episode files and writes
their action features, and with ``--segments`` around every annotated segment start; ``data`` turns
them into statistics. Every group covers contributed episodes only (``common.contributed``)
and needs the index columns of the embodiment and policy stages. The language entries
need the ``analysis`` extra (``uv sync --extra analysis``); ``--no-language`` leaves them out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index
from oopsie_backend.stats.analysis import common, from_annotations, from_data, from_index, render
from oopsie_backend.stats.analysis.language import Parser, normalize

REQUIRED = ["policy_class", "embodiment", "unified_cameras"]


def _clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_report(path: Path, group: str, episodes: int, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "group": group,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "episodes": episodes,
        "entries": entries,
    }
    path.write_text(json.dumps(_clean(report), indent=2, default=str) + "\n")
    print(f"{group}: {len(entries)} entries over {episodes} episodes -> {path}", file=sys.stderr)


def episodes_of(index: Path) -> pl.DataFrame:
    present = set(pl.read_parquet_schema(index))
    missing = [c for c in REQUIRED if c not in present]
    if missing:
        raise SystemExit(f"{index} lacks {missing}; run oopsie index embodiments, policies and camera-names first")
    columns = [c for c in common.EPISODE_COLUMNS if c in present]
    return common.contributed(read_index(index, columns, optional=common.FLAG_COLUMNS))


def returns_of(index: Path, batches: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    present = set(pl.read_parquet_schema(index))
    if "annotation_records" not in present:
        raise SystemExit(f"{index} has no annotation columns; run oopsie annotation pipeline index first")
    df = read_index(index, from_annotations.RETURNS_COLUMNS, optional=common.FLAG_COLUMNS)
    if batches:
        df = df.filter(pl.col("annotation_batch").is_in(batches))
    return from_annotations.returns_from_index(df)


def verbs_of(episodes: pl.DataFrame, cfg: dict) -> dict[str, list[str]]:
    texts = [normalize(t) for t in episodes["language_instruction"].to_list()]
    parsed = Parser(cfg).parse_all(texts)
    return {k: [s.verb for s in parsed.get(t, []) if s.verb] for k, t in zip(episodes[KEY].to_list(), texts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="group", required=True)
    for name in ("index", "annotations", "features", "data"):
        p = sub.add_parser(name)
        p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        p.add_argument("--config", type=Path, default=common.ANALYSIS_CONFIG)
        p.add_argument("--out", type=Path, required=True)
        if name in ("index", "annotations"):
            p.add_argument("--no-language", action="store_true", help="Leave out the entries that parse instructions")
        if name == "annotations":
            p.add_argument("--batch", nargs="+", metavar="NAME", help="Annotation batches to include; default all")
        if name == "features":
            p.add_argument("--segments", action="store_true", help="Also compute features around every annotated segment start")
        if name == "features":
            p.add_argument("--data-dir", type=Path, required=True)
            p.add_argument("--workers", type=int, default=8)
        if name == "data":
            p.add_argument("--features", type=Path, required=True)
    p = sub.add_parser("render", help="Write a Markdown version of each report next to it")
    p.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.group == "render":
        for path in args.reports:
            print(f"render: {path} -> {render.render_file(path)}", file=sys.stderr)
        return 0
    cfg = common.load_config(args.config)
    episodes = episodes_of(args.index)
    if args.group == "index":
        write_report(args.out, "index", episodes.height, from_index.compute(episodes, cfg, not args.no_language))
    elif args.group == "annotations":
        outcomes, segments, assigned = returns_of(args.index, args.batch)
        verbs = None if args.no_language else verbs_of(episodes.filter(pl.col(KEY).is_in(outcomes[KEY].unique().to_list())), cfg)
        entries = from_annotations.compute(episodes, outcomes, segments, cfg, verbs, assigned)
        if not args.no_language:
            keys = episodes[KEY].to_list()
            entries["47_description_words"] = from_annotations.description_words(
                outcomes.filter(pl.col(KEY).is_in(keys)), segments.filter(pl.col(KEY).is_in(keys)), cfg
            )
        write_report(args.out, "annotations", outcomes[KEY].n_unique(), entries)
    elif args.group == "features":
        segments = returns_of(args.index)[1] if args.segments else None
        feats = from_data.extract(episodes, args.data_dir, cfg, args.workers, segments)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        feats.write_parquet(args.out)
        print(f"features: {feats.height} rows -> {args.out}", file=sys.stderr)
    else:
        feats = pl.read_parquet(args.features)
        labels = common.contributor_labels(episodes)
        entries = from_data.compute(feats, episodes, labels, cfg)
        if (feats["window"] != "episode").any():
            entries["38_signatures_before_failure"] = from_data.signatures_before_failure(feats)
        entries = dict(sorted(entries.items(), key=lambda kv: int(kv[0].split("_", 1)[0])))
        write_report(args.out, "data", feats.filter(pl.col("window") == "episode").height, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
