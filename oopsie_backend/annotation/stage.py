"""Lay out the current holdings of the roster as a folder tree for a Hugging Face dataset repo.

Usage:
    uv run python -m oopsie_backend.annotation.stage \\
        --bundles-dir bundles --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir annotation_repo

Writes ``<folder>/<slot>/`` for every bundle and its current holder, holding
``eval_manifest.json`` and ``episodes/0001.h5``, ``episodes/0001_<cam>.mp4``, ...
``<folder>`` is the annotator name in lowercase ASCII with ``_`` separators and
``<slot>`` is ``batch_<n>``. A slot whose manifest already lists the bundle's
episodes for the same annotator is kept as it is; slots of bundles that moved to
another annotator are removed. A ``README.md`` at the root lists every annotator's
folder and batches and the commands to download, annotate and upload.
``annotation.pipeline publish`` uploads the tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from oopsie_backend.annotation.bundles import MANIFEST_NAME, write_bundle
from oopsie_backend.annotation.roster import SLOT_RE, Holding, Registry
from oopsie_backend.common.pools import process_pool
from oopsie_backend.common.repos import DEFAULT_DATA_DIR


def _package(job: tuple) -> tuple[str, list[str]]:
    annotator, entries, proposal, folder, data_dir, out_dir, blind = job
    target, missing = write_bundle(annotator, entries, data_dir, out_dir, proposal, blind, folder)
    return str(target), missing


def is_staged(slot_dir: Path, holding: Holding, positions: dict[int, str]) -> bool:
    manifest = slot_dir / MANIFEST_NAME
    if not manifest.is_file():
        return False
    data = json.loads(manifest.read_text())
    staged = {e["position"]: e["uid"] for e in data["episodes"]}
    return data.get("annotator") == holding.annotator and staged == positions


def stale_slots(out_dir: Path, current: set[tuple[str, str]]) -> list[Path]:
    """Staged ``<folder>/<slot>`` directories that no current holding uses."""
    return sorted(
        p for p in out_dir.glob("*/*")
        if p.is_dir() and SLOT_RE.fullmatch(p.name) and (p.parent.name, p.name) not in current
    )


def readme(rows: list[tuple[str, str, list[tuple[str, int]]]]) -> str:
    table = "\n".join(
        f"| {name} | `{folder}` | " + ", ".join(f"`{slot}` ({n})" for slot, n in slots) + " |"
        for name, folder, slots in sorted(rows)
    )
    return f"""# Oopsie annotation batches

Each annotator has one folder with one or more batches. Annotate every episode in each batch.

| Annotator | Folder | Batches (episodes) |
|---|---|---|
{table}

## Download your folder

```bash
oopsie-data annotation download
```

This asks for your full name, as listed above, and a HuggingFace token, and downloads only your folder into `oopsie-annotation/`.

## Annotate

Run this for each of your batches:

```bash
oopsie-data annotate --samples-dir oopsie-annotation/<folder>/<batch>/episodes --annotator-name "<your name>"
```

Episodes are numbered in the order they should be annotated.

## Upload your annotations

```bash
oopsie-data annotation upload
```

This opens a pull request with your annotated files on the returns repo. You can upload again at any time; later uploads replace earlier ones.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundles-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("annotation_repo"))
    parser.add_argument(
        "--keep-annotations",
        action="store_true",
        help="Keep existing annotations, source data and episode ids in the copies",
    )
    parser.add_argument("--only", nargs="+", metavar="ANNOTATOR", help="Stage only these annotators")
    parser.add_argument("--workers", type=int, default=4, help="Bundles written in parallel")
    args = parser.parse_args(argv)

    registry = Registry(args.bundles_dir)
    current = registry.current()
    if not current:
        print(f"error: no bundles in {args.bundles_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in stale_slots(args.out_dir, {(h.folder, h.slot) for h in current}):
        shutil.rmtree(path)
        if not any(path.parent.iterdir()):
            path.parent.rmdir()
        print(f"removed {path}: its bundle moved", file=sys.stderr)

    rows: dict[str, tuple[str, list[tuple[str, int]]]] = {}
    jobs = []
    for h in current:
        bundle = registry.bundles[h.bundle]
        rows.setdefault(h.annotator, (h.folder, []))[1].append((h.slot, len(bundle["episodes"])))
        if args.only and h.annotator not in args.only:
            continue
        if is_staged(args.out_dir / h.folder / h.slot, h, registry.positions(h)):
            continue
        jobs.append((
            h.annotator, bundle["episodes"], bundle.get("proposal"), f"{h.folder}/{h.slot}",
            args.data_dir, args.out_dir, not args.keep_annotations,
        ))
    (args.out_dir / "README.md").write_text(
        readme([
            (name, folder, sorted(slots, key=lambda s: int(SLOT_RE.fullmatch(s[0]).group(1))))
            for name, (folder, slots) in rows.items()
        ])
    )

    failed = 0
    with process_pool(max_workers=args.workers) as pool:
        for target, missing in pool.map(_package, jobs):
            for msg in missing:
                print(f"  {target}: {msg}", file=sys.stderr)
            failed += len(missing)
    print(
        f"{len(jobs)} bundle(s) staged in {args.out_dir}, {len(current) - len(jobs)} already staged or skipped, "
        f"{failed} episode(s) not packaged",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
