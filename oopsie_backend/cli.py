"""``oopsie <stage> <command> [args]``: one entry point for every pipeline command.

Each command is a module with its own argument parser; ``oopsie <stage> <command> --help``
shows its options. ``oopsie`` alone lists the commands.
"""

from __future__ import annotations

import runpy
import sys

COMMANDS: dict[str, dict[str, str]] = {
    "ingest": {
        "sync": "ingest.sync",
        "download": "ingest.download",
        "upload-times": "ingest.upload_times",
        "release": "ingest.release",
    },
    "index": {"build": "index.build", "camera-names": "index.camera_names", "camera-sheet": "index.camera_sheet"},
    "qa": {
        "run": "qa.run",
        "render-report": "qa.render_report",
        "serve-report": "qa.serve_report",
        "select-spot-checks": "qa.select_spot_checks",
        "merge-notes": "qa.merge_notes",
        "render-lab-feedback": "qa.render_lab_feedback",
    },
    "stats": {
        "compute-local": "stats.compute_local",
        "compute-hf": "stats.compute_hf",
        "render-public": "stats.render_public",
        "render-internal": "stats.render_internal",
    },
    "splits": {
        "propose": "splits.propose",
        "record": "splits.record",
        "sample-train": "splits.sample_train",
        "package": "splits.package",
    },
    "annotation": {
        "assign": "annotation.assign",
        "package": "annotation.package",
        "stage": "annotation.stage",
        "collect": "annotation.collect_returns",
        "pipeline": "annotation.pipeline",
    },
    "pipeline": {"run": "pipeline.run"},
    "training": {"materialize": "training.build_dataset"},
    "export": {"to-rlds": "export.to_rlds"},
    "convert": {
        "imports": "conversion.imports",
        "ar-aloha": "conversion.convert_ar_aloha_data",
        "rlds": "conversion.convert_rlds_to_hdf5",
        "roboarena": "conversion.convert_roboarena_data",
        "soar": "conversion.convert_soar_data",
    },
    "tools": {
        "inspect-hdf5": "tools.inspect_hdf5",
        "visualize": "tools.visualize_episodes",
        "parquet-html": "tools.parquet_html",
    },
}


def usage() -> str:
    lines = ["usage: oopsie <stage> <command> [args]", ""]
    for stage, commands in COMMANDS.items():
        lines.append(f"  {stage:<11} " + ", ".join(commands))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in COMMANDS or argv[1] not in COMMANDS[argv[0]]:
        print(usage(), file=sys.stderr)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    stage, command, rest = argv[0], argv[1], argv[2:]
    sys.argv = [f"oopsie {stage} {command}", *rest]
    try:
        runpy.run_module(f"oopsie_backend.{COMMANDS[stage][command]}", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
