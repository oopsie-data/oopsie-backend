"""Run the Hugging Face annotation pipeline from one config file.

Usage:
    uv run python -m oopsie_backend.annotation.pipeline \\
        --config configs/annotation_pipeline.yaml <command>

Commands, in pipeline order:
  stage     (LS6) package every batch into ``stage_dir`` as ``<folder>/<batch>/``
  publish   create the private dataset repo if needed and upload ``stage_dir``
  pull      download annotated h5s from the repo's pull requests to ``returns_dir``
  collect   add pulled annotations to each batch's store and write its summary
  status    returned vs assigned annotations per annotator and batch
  index     write assignment and collected annotations to the episode index
  release   (LS6) write source episodes with collected annotations to each batch's ``release_dir``

Relative paths in the config resolve against ``<data_dir>/annotation``, next to the
mirror's org folder; ``--data-dir`` and ``--index`` override the config's values.
``pull`` and ``collect`` can be re-run as returns arrive; both skip what they
already have. When the config names an ``index``, ``collect`` also runs ``index``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

from oopsie_backend.annotation import collect_returns
from oopsie_backend.annotation import stage as stage_repo
from oopsie_backend.annotation.collect import (
    current_annotations,
    load_assignment,
    read_store,
)
from oopsie_backend.annotation.columns import ANNOTATION_COLUMNS, annotation_frame
from oopsie_backend.annotation.hub import (
    PipelineConfig,
    annotator_folders,
    load_config,
    pull_returns,
    returned_batch_dirs,
)
from oopsie_backend.index.store import known_only, update_columns


def stage(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    argv = [arg for b in cfg.batches for arg in ("--batch", f"{b.name}={b.assignment_dir}")]
    argv += [
        "--data-dir", str(cfg.data_dir),
        "--out-dir", str(cfg.stage_dir),
        "--repo-id", cfg.repo_id,
        "--workers", str(args.workers),
    ]
    if args.only:
        argv += ["--only", *args.only]
    return stage_repo.main(argv)


def publish(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    if not (cfg.stage_dir / "README.md").is_file():
        print(f"error: {cfg.stage_dir} is not staged; run stage first", file=sys.stderr)
        return 1
    api = HfApi()
    api.create_repo(cfg.repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(cfg.repo_id, repo_type="dataset").private:
        print(f"error: {cfg.repo_id} is public; make it private before uploading", file=sys.stderr)
        return 1
    api.upload_large_folder(cfg.repo_id, cfg.stage_dir, repo_type="dataset", num_workers=args.workers)
    print(f"uploaded {cfg.stage_dir} to https://huggingface.co/datasets/{cfg.repo_id}", file=sys.stderr)
    return 0


def pull(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    folders = set(annotator_folders(cfg).values())
    rows = pull_returns(HfApi(), cfg, folders, close=args.close)
    for r in rows:
        print(f"  PR #{r['num']} {r['author']} {r['folder'] or '-'} ({r['files']} files): {r['result']}", file=sys.stderr)
    rejected = sum(r["result"].startswith("rejected") for r in rows)
    downloaded = sum(r["result"].startswith("downloaded") for r in rows)
    print(f"{len(rows)} pull request(s): {downloaded} downloaded, {rejected} rejected", file=sys.stderr)
    return 1 if rejected else 0


def collect(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    rc = 0
    for b in cfg.batches:
        dirs = returned_batch_dirs(cfg, b.name)
        print(f"{b.name}: {len(dirs)} returned folder(s)", file=sys.stderr)
        argv = [*map(str, dirs), "--assignment-dir", str(b.assignment_dir), "--store", str(b.store)]
        rc |= collect_returns.main(argv)
    if cfg.index is not None:
        rc |= index(cfg, args)
    return rc


def missing_assignments(cfg: PipelineConfig) -> list[Path]:
    return [b.assignment_dir for b in cfg.batches if not load_assignment(b.assignment_dir)]


def index(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    if cfg.index is None:
        print("error: the config names no index", file=sys.stderr)
        return 1
    missing = missing_assignments(cfg)
    if missing:
        print(f"error: no annotator lists in {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    frame = annotation_frame([(b.name, b.assignment_dir, b.store) for b in cfg.batches])
    frame = known_only(frame, cfg.index, "assigned")
    update_columns(cfg.index, frame, ANNOTATION_COLUMNS, clear_missing=True)
    print(f"index: annotation columns written for {frame.height} episode(s) in {cfg.index}", file=sys.stderr)
    return 0


def status(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    missing = missing_assignments(cfg)
    if missing:
        print(f"error: no annotator lists in {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    folders = annotator_folders(cfg)
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for b in cfg.batches:
        assignment = load_assignment(b.assignment_dir)
        returned = {
            (r["uid"], r["bundle"]) for rs in current_annotations(read_store(b.store)).values() for r in rs
        }
        for name in folders:
            uids = [uid for uid, m in assignment.items() if name in m["assigned"]]
            counts[(name, b.name)] = (sum((uid, name) in returned for uid in uids), len(uids))
    width = max(map(len, folders))
    print(f"{'annotator':<{width}}  " + "  ".join(f"{b.name:>11}" for b in cfg.batches))
    for name in sorted(folders):
        cells = [f"{counts[(name, b.name)][0]:>5}/{counts[(name, b.name)][1]:<5}" for b in cfg.batches]
        print(f"{name:<{width}}  " + "  ".join(cells))
    for b in cfg.batches:
        done = sum(counts[(n, b.name)][0] for n in folders)
        total = sum(counts[(n, b.name)][1] for n in folders)
        print(f"{b.name}: {done}/{total} annotations returned")
    prs = cfg.returns_dir / "pull_requests.json"
    if prs.is_file():
        rejected = [r for r in json.loads(prs.read_text()) if r["result"].startswith("rejected")]
        for r in rejected:
            print(f"PR #{r['num']} by {r['author']} {r['result']}")
    return 0


def release(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    rc = 0
    for b in cfg.batches:
        argv = [
            "--assignment-dir", str(b.assignment_dir),
            "--store", str(b.store),
            "--apply-to", str(b.release_dir),
            "--data-dir", str(cfg.data_dir),
        ]
        rc |= collect_returns.main(argv)
    return rc


COMMANDS = {
    "stage": stage,
    "publish": publish,
    "pull": pull,
    "collect": collect,
    "status": status,
    "index": index,
    "release": release,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/annotation_pipeline.yaml"))
    parser.add_argument("--data-dir", type=Path, help="Mirror root; annotation files live in its annotation/ folder")
    parser.add_argument("--index", type=Path, help="Episode index")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--only", nargs="+", metavar="ANNOTATOR")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("publish")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("pull")
    p.add_argument("--close", action="store_true", help="Close open pull requests after downloading them")
    for name in ("collect", "status", "index", "release"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    return COMMANDS[args.command](load_config(args.config, args.data_dir, args.index), args)


if __name__ == "__main__":
    raise SystemExit(main())
