"""Lay out annotation batches as a folder tree for a Hugging Face dataset repo.

Usage:
    uv run python -m oopsie_backend.annotation.stage \\
        --batch batch_1=eval_assignment --batch batch_2=train_assignment \\
        --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir annotation_repo \\
        --repo-id OopsieData-Submissions/oopsie-annotation

Writes ``<folder>/<batch>/`` per annotator and batch, each holding
``eval_manifest.json`` and ``episodes/0001.h5``, ``episodes/0001_<cam>.mp4``, ...
``<folder>`` is the annotator name in lowercase ASCII with ``_`` separators.
A ``README.md`` at the root lists every annotator's folder and the commands
to download, annotate and upload. Upload the tree with

    hf upload-large-folder <repo-id> annotation_repo --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from oopsie_backend.annotation.bundles import write_bundle
from oopsie_backend.annotation.hub import folder_name
from oopsie_backend.common.repos import DEFAULT_DATA_DIR


def _batch(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not re.fullmatch(r"[a-z0-9_]+", name):
        raise argparse.ArgumentTypeError(f"expected NAME=ASSIGNMENT_DIR with NAME in [a-z0-9_], got {spec!r}")
    return name, Path(path)


def _package(job: tuple) -> tuple[str, list[str]]:
    path, folder, data_dir, out_dir, blind = job
    lists = json.loads(Path(path).read_text())
    target, missing = write_bundle(
        lists["annotator"], lists["episodes"], data_dir, out_dir, lists.get("proposal"), blind, folder
    )
    return str(target), missing


def readme(repo_id: str, batches: list[str], folders: dict[str, str], counts: dict[tuple[str, str], int]) -> str:
    rows = "\n".join(
        f"| {name} | `{folder}` | " + " | ".join(str(counts.get((name, b), 0)) for b in batches) + " |"
        for name, folder in sorted(folders.items())
    )
    annotate = "\n".join(
        f'oopsie-data annotate --samples-dir oopsie-annotation/<folder>/{b}/episodes --annotator-name "<your name>"'
        for b in batches
    )
    return f"""# Oopsie annotation batches

Each annotator has one folder with {len(batches)} batches ({", ".join(batches)}). Annotate every episode in each batch.

| Annotator | Folder | {" | ".join(batches)} |
|---|---|{"---:|" * len(batches)}
{rows}

## Download your folder

```bash
hf download {repo_id} --repo-type dataset --include "<folder>/*" --local-dir oopsie-annotation
```

## Annotate

```bash
{annotate}
```

Episodes are numbered in the order they should be annotated.

## Upload your annotations

```bash
hf upload {repo_id} oopsie-annotation/<folder> <folder> --repo-type dataset --include "*.h5" --create-pr
```

This opens a pull request with your annotated files. You can upload again at any time; later uploads replace earlier ones.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch", action="append", required=True, type=_batch, metavar="NAME=ASSIGNMENT_DIR",
        help="Batch folder name and the assign_eval_annotators output it is built from; repeatable",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("annotation_repo"))
    parser.add_argument("--repo-id", default="OopsieData-Submissions/oopsie-annotation")
    parser.add_argument(
        "--keep-annotations",
        action="store_true",
        help="Keep existing annotations, source data and episode ids in the copies",
    )
    parser.add_argument("--only", nargs="+", metavar="ANNOTATOR", help="Stage only these annotators")
    parser.add_argument("--workers", type=int, default=4, help="Bundles written in parallel")
    args = parser.parse_args(argv)

    batch_names = [name for name, _ in args.batch]
    if len(set(batch_names)) != len(batch_names):
        print(f"error: repeated batch names {batch_names}", file=sys.stderr)
        return 1

    folders: dict[str, str] = {}
    counts: dict[tuple[str, str], int] = {}
    jobs = []
    for batch, assignment_dir in args.batch:
        lists = [p for p in sorted(assignment_dir.glob("*.json")) if p.name != "assignment_summary.json"]
        if not lists:
            print(f"error: no annotator lists in {assignment_dir}", file=sys.stderr)
            return 1
        for path in lists:
            data = json.loads(path.read_text())
            name = data["annotator"]
            folders[name] = folder_name(name)
            counts[(name, batch)] = len(data["episodes"])
            if args.only and name not in args.only:
                continue
            jobs.append((path, f"{folders[name]}/{batch}", args.data_dir, args.out_dir, not args.keep_annotations))

    taken: dict[str, str] = {}
    for name, folder in folders.items():
        if not folder or folder in taken:
            print(f"error: annotators {taken.get(folder)!r} and {name!r} map to folder {folder!r}", file=sys.stderr)
            return 1
        taken[folder] = name

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.md").write_text(readme(args.repo_id, batch_names, folders, counts))

    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for target, missing in pool.map(_package, jobs):
            for msg in missing:
                print(f"  {target}: {msg}", file=sys.stderr)
            failed += len(missing)
    print(f"{len(jobs)} bundle(s) in {args.out_dir}, {failed} episode(s) not packaged", file=sys.stderr)
    print(f"upload with: hf upload-large-folder {args.repo_id} {args.out_dir} --repo-type dataset", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
