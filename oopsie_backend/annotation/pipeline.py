"""Run the Hugging Face annotation pipeline from one config file.

Usage:
    uv run python -m oopsie_backend.annotation.pipeline \\
        --config configs/annotation_pipeline.yaml <command>

Commands, in pipeline order:
  bundles   list every bundle, its kind, size and holders
  add       build a new train bundle for an annotator
  reassign  give a bundle to another annotator
  stage     (LS6) package every current holding into ``stage_dir`` as ``<folder>/<slot>/``
  publish   create the private batch and returns repos if needed, upload ``stage_dir`` and remove moved bundles
  pull      download annotated h5s from the returns repo's pull requests to ``returns_dir``
  verify    check that every pulled h5 matches its bundle, and its watched videos the index episode's
  collect   add verified annotations to the store of their bundle kind and write its summary
  status    returned vs assigned annotations per annotator and batch
  index     write assignments and collected annotations, with their segments, to the episode index
  release   (LS6) write source episodes with collected annotations to each kind's ``release_dir``

Relative paths in the config resolve against ``<data_dir>/annotation``, next to the
mirror's org folder; ``--data-dir`` and ``--index`` override the config's values.
``pull`` and ``collect`` can be re-run as returns arrive; both skip what they
already have. ``collect`` needs the ``index``; it runs ``verify`` first and then ``index``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

from oopsie_backend.annotation import stage as stage_repo
from oopsie_backend.annotation.collect import (
    apply_to_copies,
    coverage_report,
    current_annotations,
    merge_into_store,
    read_annotated,
    read_store,
)
from oopsie_backend.annotation.columns import ANNOTATION_COLUMNS, annotation_frame
from oopsie_backend.annotation.hub import PipelineConfig, annotator_slots, file_ids, load_config, pull_returns
from oopsie_backend.annotation.roster import SLOT_RE, TRAIN, Registry
from oopsie_backend.annotation.train_bundles import (
    TRAIN_ANNOTATORS,
    TrainPool,
    add_train_bundle,
    record_train_target,
    train_held,
)
from oopsie_backend.annotation.verify import index_by_uid, verify_returns, write_report
from oopsie_backend.annotation.videos import CACHE_NAME as VIDEOS_CACHE, bundle_videos, returned_position
from oopsie_backend.common.repos import annotation_dir
from oopsie_backend.index.store import KEY, known_only, read_index, update_columns
from oopsie_backend.splits.stages import load_split_config


def _registry(cfg: PipelineConfig) -> Registry | None:
    registry = cfg.registry()
    if not registry.exists:
        print(f"error: no roster in {cfg.bundles_dir}; run the annotation pipeline target first", file=sys.stderr)
        return None
    return registry


def bundles(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    history: dict[str, list[str]] = defaultdict(list)
    for h in registry.holdings:
        history[h.bundle].append(h.annotator)
    for h in registry.current():
        b = registry.bundles[h.bundle]
        former = history[h.bundle][:-1]
        extra = f"  (formerly {', '.join(former)})" if former else ""
        cal = f", {b['calibration']} calibration" if b.get("calibration") else ""
        print(f"{h.bundle}  {b['kind']:<5}  {len(b['episodes']):>4} episodes{cal}  {h.annotator} / {h.slot}{extra}")
    return 0


def add(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    if cfg.index is None:
        print("error: building a train bundle needs an index", file=sys.stderr)
        return 1
    assignments = load_split_config(str(args.split_config)).get("assignments", {})
    train = assignments.get("train", {})
    pool = TrainPool(cfg.index, annotation_dir(cfg.data_dir) / "eval_proposal.json")
    targets = annotation_dir(cfg.data_dir) / train.get("annotators_file", TRAIN_ANNOTATORS)
    returned = {uid for s in cfg.stores if s.kind == TRAIN for uid in current_annotations(read_store(s.store))}
    for _ in range(args.count):
        try:
            h = add_train_bundle(
                registry, pool, args.annotator,
                size=args.size or train.get("bundle_size", 200),
                calibration=train.get("calibration_episodes", 20),
                returned=returned, seed=assignments.get("seed", 0), note=args.note,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        b = registry.bundles[h.bundle]
        record_train_target(targets, args.annotator, train_held(registry, args.annotator))
        print(f"{h.bundle}: {len(b['episodes'])} train episodes ({b['calibration']} calibration) -> {h.annotator}, {h.slot}")
    print(f"{targets.name}: {args.annotator}: {train_held(registry, args.annotator)}", file=sys.stderr)
    print("the next pipeline run publishes it; run stage and publish to send it now", file=sys.stderr)
    return 0


def reassign(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    try:
        before = registry.holder(args.bundle) if args.bundle in registry.bundles else None
        h = registry.reassign(args.bundle, args.to, args.note)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{h.bundle}: {before.annotator} / {before.slot} -> {h.annotator} / {h.slot}")
    print("the next pipeline run publishes it; run stage and publish to move it now", file=sys.stderr)
    return 0


def stage(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    argv = [
        "--bundles-dir", str(cfg.bundles_dir),
        "--data-dir", str(cfg.data_dir),
        "--out-dir", str(cfg.stage_dir),
        "--workers", str(args.workers),
    ]
    if args.only:
        argv += ["--only", *args.only]
    return stage_repo.main(argv)


def publish(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    if not (cfg.stage_dir / "README.md").is_file():
        print(f"error: {cfg.stage_dir} is not staged; run stage first", file=sys.stderr)
        return 1
    api = HfApi()
    for repo in (cfg.repo_id, cfg.returns_repo_id):
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type="dataset").private:
            print(f"error: {repo} is public; make it private before uploading", file=sys.stderr)
            return 1
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(cfg.repo_id, cfg.stage_dir, repo_type="dataset", num_workers=args.workers)
    else:
        # huggingface_hub 2 commits large folders in batches and resumes in upload_folder itself.
        api.upload_folder(repo_id=cfg.repo_id, folder_path=cfg.stage_dir, repo_type="dataset")
    current = {(h.folder, h.slot) for h in registry.current()}
    published = {tuple(p.split("/")[:2]) for p in file_ids(api, cfg.repo_id, None) if p.count("/") >= 2}
    stale = sorted(s for s in published if SLOT_RE.fullmatch(s[1]) and s not in current)
    if stale:
        api.create_commit(
            cfg.repo_id,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f"{f}/{s}/", is_folder=True) for f, s in stale],
            commit_message=f"Remove {len(stale)} reassigned batch(es)",
        )
        print(f"removed {', '.join(f'{f}/{s}' for f, s in stale)}: their bundles moved", file=sys.stderr)
    print(f"uploaded {cfg.stage_dir} to https://huggingface.co/datasets/{cfg.repo_id}", file=sys.stderr)
    return 0


def pull(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    rows = pull_returns(HfApi(), cfg, annotator_slots(registry), close=args.close)
    for r in rows:
        print(f"  PR #{r['num']} {r['author']} {r['folder'] or '-'} ({r['files']} files): {r['result']}", file=sys.stderr)
    rejected = sum(r["result"].startswith("rejected") for r in rows)
    skipped = sum(r["result"].startswith("skipped") for r in rows)
    downloaded = sum(r["result"].startswith("downloaded") for r in rows)
    print(f"{len(rows)} pull request(s): {downloaded} downloaded, {rejected} rejected, {skipped} skipped", file=sys.stderr)
    return 1 if rejected or skipped else 0


def _verified(cfg: PipelineConfig, registry: Registry) -> tuple[dict, dict, int] | None:
    """``(kind -> annotator -> matched returned files, path -> watched videos, number rejected)``.

    Writes ``<returns_dir>/verification.json``. Unmatched files are logged and left out.
    """
    if cfg.index is None:
        print("error: verifying returns needs an index", file=sys.stderr)
        return None
    checks = verify_returns(cfg, registry, index_by_uid(cfg.index))
    write_report(cfg.returns_dir / "verification.json", checks)
    out: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[str, dict[str, str]] = {}
    rejected = unmatched = 0
    for c in checks:
        if c.problems:
            rejected += 1
            print(f"  rejected {c.path}: {'; '.join(c.problems)}", file=sys.stderr)
        elif c.unmatched:
            unmatched += 1
            print(f"  unmatched {c.path}: {c.unmatched}", file=sys.stderr)
        else:
            out[c.kind][c.annotator].append(c.path)
            seen[str(c.path)] = c.seen
    for s in cfg.stores:
        n = sum(map(len, out[s.kind].values()))
        print(f"{s.kind}: {n} returned file(s) verified", file=sys.stderr)
    print(f"{rejected}/{len(checks)} returned file(s) rejected, {unmatched} unmatched", file=sys.stderr)
    return out, seen, rejected


def verify(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    result = registry and _verified(cfg, registry)
    return 1 if not result or result[2] else 0


def collect(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    result = registry and _verified(cfg, registry)
    if not result:
        return 1
    verified, seen, _ = result
    for s in cfg.stores:
        assignment = registry.assignment(s.kind)
        records, problems = [], []
        for annotator, files in sorted(verified[s.kind].items()):
            recs, probs = read_annotated(files, annotator, assignment)
            records += recs
            problems += probs
        for r in records:
            r["videos"] = seen[r["source_file"]]
        for msg in problems:
            print(f"  skipped {msg}", file=sys.stderr)
        added, present = merge_into_store(s.store, records)
        report = coverage_report(read_store(s.store), assignment)
        summary = s.store.with_name(s.store.stem + "_summary.json")
        summary.write_text(json.dumps(report, indent=2))
        print(
            f"{s.kind}: {len(records)} annotation(s) read, {added} added, {present} already in {s.store}; "
            f"returned {report['annotations_returned']}/{report['annotations_assigned']}, see {summary}",
            file=sys.stderr,
        )
    return 0 if args.skip_index else index(cfg, args)


def index(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    if cfg.index is None:
        print("error: the config names no index", file=sys.stderr)
        return 1
    instructions = dict(read_index(cfg.index, ["language_instruction"]).select(KEY, "language_instruction").iter_rows())
    without = [r["source_file"] for s in cfg.stores for r in read_store(s.store) if not r.get("videos")]
    slots = {found[0] for f in without if (found := returned_position(f))}
    videos = bundle_videos(cfg.repo_id, slots, cfg.returns_dir / VIDEOS_CACHE) if slots else {}
    frame = annotation_frame([(s.kind, registry.assignment(s.kind), s.store) for s in cfg.stores], instructions, videos)
    frame = known_only(frame, cfg.index, "assigned")
    update_columns(cfg.index, frame, ANNOTATION_COLUMNS, clear_missing=True)
    print(f"index: annotation columns written for {frame.height} episode(s) in {cfg.index}", file=sys.stderr)
    return 0


def status(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    returned: set[tuple[str, str]] = set()
    for s in cfg.stores:
        returned |= {(r["uid"], r["bundle"]) for rs in current_annotations(read_store(s.store)).values() for r in rs}
    rows: dict[str, list[str]] = defaultdict(list)
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for h in sorted(registry.current(), key=lambda h: int(SLOT_RE.fullmatch(h.slot).group(1))):
        kind = registry.bundles[h.bundle]["kind"]
        uids = list(registry.positions(h).values())
        done = sum((uid, h.annotator) in returned for uid in uids)
        rows[h.annotator].append(f"{h.slot} {kind:<5} {done:>4}/{len(uids):<4}")
        totals[kind][0] += done
        totals[kind][1] += len(uids)
    if not rows:
        print(f"no bundles in {cfg.bundles_dir}")
    width = max(map(len, rows), default=0)
    for name in sorted(rows):
        print(f"{name:<{width}}  " + "   ".join(rows[name]))
    for kind, (done, total) in sorted(totals.items()):
        print(f"{kind}: {done}/{total} annotations returned")
    prs = cfg.returns_dir / "pull_requests.json"
    if prs.is_file():
        rejected = [r for r in json.loads(prs.read_text()) if r["result"].startswith("rejected")]
        for r in rejected:
            print(f"PR #{r['num']} by {r['author']} {r['result']}")
    return 0


def release(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    registry = _registry(cfg)
    if registry is None:
        return 1
    for s in cfg.stores:
        apply_to_copies(read_store(s.store), registry.assignment(s.kind), cfg.data_dir, s.release_dir)
    return 0


COMMANDS = {
    "bundles": bundles,
    "add": add,
    "reassign": reassign,
    "stage": stage,
    "publish": publish,
    "pull": pull,
    "verify": verify,
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
    sub.add_parser("bundles")
    p = sub.add_parser("add")
    p.add_argument("--annotator", required=True, help="Name as listed in annotators.txt")
    p.add_argument("--count", type=int, default=1, help="Bundles to add")
    p.add_argument("--size", type=int, help="Episodes per bundle; default assignments.train.bundle_size")
    p.add_argument("--split-config", type=Path, default=Path("configs/eval_split.yaml"))
    p.add_argument("--note", default="requested")
    p = sub.add_parser("reassign")
    p.add_argument("bundle", help="Bundle id, as listed by the bundles command")
    p.add_argument("--to", required=True, metavar="ANNOTATOR")
    p.add_argument("--note", default="")
    p = sub.add_parser("stage")
    p.add_argument("--only", nargs="+", metavar="ANNOTATOR")
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("publish")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("pull")
    p.add_argument("--close", action="store_true", help="Close open pull requests after downloading them")
    p = sub.add_parser("collect")
    p.add_argument("--skip-index", action="store_true", help="Update the stores only; do not write the index")
    for name in ("verify", "status", "index", "release"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    return COMMANDS[args.command](load_config(args.config, args.data_dir, args.index), args)


if __name__ == "__main__":
    raise SystemExit(main())
