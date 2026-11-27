"""Convert the local imports of ``configs/imports.yaml`` into ``<data root>/imports/<name>``.

Usage:
    uv run oopsie convert imports --data-dir /corral/utexas/IRI26024/oopsiedata [--only NAME]

A ``converter`` entry is converted into ``.<name>.partial`` in ``--work-dir``
(default: the imports folder) and moved to ``imports/<name>`` once its converter finishes; rerunning after a failure resumes,
since converters skip episodes they have already written. Each converted dataset
keeps a ``<name>.inputs`` record of its converter, source and arguments. If an
entry changes later, the run fails instead of converting again; delete the
folder and its record to reconvert. ``hub`` entries are only checked to be
present in the mirror.

The next index build picks up the converted datasets as external datasets, and
the QA stage samples them like every other dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from oopsie_backend.common.h5 import find_h5
from oopsie_backend.common.repos import (
    DEFAULT_DATA_DIR,
    IMPORTS_CONFIG,
    imports_dir,
    load_imports,
    org_dir,
)
from oopsie_backend.conversion.registry import CONVERTERS
from oopsie_backend.pipeline.registry import stat_fingerprint, write_once


def convert_entry(name: str, entry: dict, data_dir: Path, work_dir: Path | None = None) -> str:
    """Convert one ``converter`` entry once; returns the fingerprint of its folder."""
    converter = CONVERTERS.get(entry["converter"])
    if converter is None:
        raise ValueError(f"{name}: unknown converter {entry['converter']!r}; known: {sorted(CONVERTERS)}")
    root = imports_dir(data_dir)
    out = root / name
    args = [str(a) for a in entry.get("args") or []]
    inputs = json.dumps({"converter": entry["converter"], "source": str(entry["source"]), "args": args}, sort_keys=True)

    def create() -> None:
        partial = (work_dir or root) / f".{name}.partial"
        partial.mkdir(parents=True, exist_ok=True)
        print(f"converting {name} with {entry['converter']} from {entry['source']}", file=sys.stderr)
        converter.main([converter.source_flag, str(entry["source"]), "--output-dir", str(partial), *args])
        if not find_h5(partial):
            raise RuntimeError(f"{name}: the converter wrote no episodes into {partial}")
        shutil.move(partial, out)

    return write_once(out, inputs, create, stat_fingerprint)


def check_hub_entry(name: str, entry: dict, data_dir: Path) -> str:
    folder = org_dir(data_dir) / entry["hub"].split("/", 1)[1]
    if not folder.is_dir():
        print(f"{name}: {entry['hub']} is not in the mirror at {folder}; run the sync", file=sys.stderr)
    return "present" if folder.is_dir() else "absent"


def import_datasets(
    data_dir: Path, only: set[str] | None = None, config: Path = IMPORTS_CONFIG, work_dir: Path | None = None
) -> dict[str, str]:
    """Fingerprint per imported dataset, converting local imports that are missing."""
    out = {}
    for name, entry in load_imports(config).items():
        if only and name not in only:
            continue
        if entry.get("converter"):
            out[name] = convert_entry(name, entry, data_dir, work_dir)
        else:
            out[name] = check_hub_entry(name, entry, data_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=IMPORTS_CONFIG)
    parser.add_argument("--only", action="append", default=None, metavar="NAME")
    parser.add_argument("--work-dir", type=Path, default=None, help="Convert here first; default: the imports folder")
    args = parser.parse_args(argv)
    unknown = set(args.only or []) - set(load_imports(args.config))
    if unknown:
        parser.error(f"not in {args.config}: {sorted(unknown)}")
    for name, fingerprint in import_datasets(args.data_dir, set(args.only or []), args.config, args.work_dir).items():
        print(f"{name}: {fingerprint}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
