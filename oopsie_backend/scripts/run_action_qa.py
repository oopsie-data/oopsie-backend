"""Run the action-space data quality checks (oopsie_backend.utils.action_qa) against
real per-timestep actions/* and observations/robot_states/* arrays.

This is a separate, opt-in pass from compute_local_stats.py: it reads full arrays
(not just HDF5 attrs), which costs much more per file, so it is not run as part of
the nightly stats job. See reports/data_quality_exceptions.md for what each check
means and configs/stats_config.yaml's action_qa block for thresholds.

Usage:
    uv run python -m oopsie_backend.scripts.run_action_qa \\
        --data-dir /media/arpit/ext_drive1/oopsiedata

    # Targeted re-check of specific datasets, no per-dataset cap:
    uv run python -m oopsie_backend.scripts.run_action_qa \\
        --data-dir /media/arpit/ext_drive1/oopsiedata \\
        --only-dataset OopsieData-Submissions/SomeLab --limit 0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py

from oopsie_backend.scripts.compute_local_stats import DEFAULT_DATA_DIR, discover_episodes
from oopsie_backend.utils.action_qa import run_action_qa_on_h5
from oopsie_backend.utils.episode_stats import REPORTS_DIR, STATS_EXCLUDED_DATASETS, now_timestamps, write_stats_json
from oopsie_backend.utils.stats_config import CONFIG

DEFAULT_OUTPUT = REPORTS_DIR / "action_qa.json"
MAX_FLAGGED_DETAIL = 500


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max episodes checked per dataset, first N (default: "
            "action_qa.sampling.default_episode_limit from stats_config.yaml). "
            "0 means unlimited."
        ),
    )
    parser.add_argument(
        "--max-total-episodes",
        type=int,
        default=None,
        help="Global cap across all datasets combined (default: unlimited).",
    )
    parser.add_argument(
        "--only-dataset",
        action="append",
        default=None,
        metavar="REPO",
        help="Restrict to this dataset repo (e.g. OopsieData-Submissions/SomeLab). Repeatable.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    qa_config = CONFIG["action_qa"]
    per_dataset_limit = args.limit if args.limit is not None else qa_config["sampling"]["default_episode_limit"]
    per_dataset_limit = None if per_dataset_limit in (0, None) else per_dataset_limit
    only_dataset = set(args.only_dataset) if args.only_dataset else None

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    episodes = discover_episodes(data_dir)
    by_dataset: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for dataset_repo, repo_path, path in episodes:
        if dataset_repo in STATS_EXCLUDED_DATASETS:
            continue
        if only_dataset is not None and dataset_repo not in only_dataset:
            continue
        by_dataset[dataset_repo].append((repo_path, path))

    if not by_dataset:
        print("error: no matching .h5 files found", file=sys.stderr)
        return 1

    planned = {
        ds: (files if per_dataset_limit is None else files[:per_dataset_limit])
        for ds, files in by_dataset.items()
    }
    total_planned = sum(len(files) for files in planned.values())
    if args.max_total_episodes is not None:
        total_planned = min(total_planned, args.max_total_episodes)
    _log(f"Will check up to {total_planned} episode(s) across {len(planned)} dataset(s)")

    totals = Counter()
    per_dataset_stats: dict[str, dict] = {}
    flagged_episodes: list[dict] = []
    checked_total = 0

    for index, dataset_repo in enumerate(sorted(planned), 1):
        if args.max_total_episodes is not None and checked_total >= args.max_total_episodes:
            break
        files = planned[dataset_repo]
        if args.max_total_episodes is not None:
            remaining = args.max_total_episodes - checked_total
            files = files[:remaining]

        ds_totals = Counter()
        ds_tests = {
            "duplicate_pairs": Counter(),
            "scale_consistency": Counter(),
            "tracking_residual": Counter(),
            "delta_detection": Counter(),
            "length_mismatch": Counter(),
        }
        delta_flagged_keys: Counter = Counter()

        _log(f"[{index}/{len(planned)}] {dataset_repo}: {len(files)} episode(s)")
        for i, (repo_path, path) in enumerate(files, 1):
            checked_total += 1
            ds_totals["episodes_checked"] += 1
            try:
                with h5py.File(path, "r") as f:
                    result = run_action_qa_on_h5(f, qa_config)
            except Exception as exc:
                ds_totals["errors"] += 1
                if args.verbose:
                    _log(f"  [{i}/{len(files)}] {repo_path}: ERROR ({exc})")
                continue

            if result is None or not result["findings"]:
                continue

            tests_hit = {finding["test"] for finding in result["findings"]}
            for test in tests_hit:
                ds_tests[test]["flagged_episodes"] += 1
            for finding in result["findings"]:
                if finding["test"] == "duplicate_pairs":
                    ds_tests["duplicate_pairs"]["flagged_pairs"] += 1
                elif finding["test"] == "scale_consistency":
                    ds_tests["scale_consistency"]["flagged_keys"] += 1
                elif finding["test"] == "delta_detection":
                    delta_flagged_keys[finding["key"]] += 1

            ds_totals["episodes_flagged"] += 1
            episode_id = path.stem
            flagged_episodes.append(
                {
                    "dataset_repo": dataset_repo,
                    "repo_path": repo_path,
                    "episode_id": episode_id,
                    "findings": result["findings"],
                }
            )
            if args.verbose:
                _log(f"  [{i}/{len(files)}] {repo_path}: FLAGGED ({sorted(tests_hit)})")

        per_dataset_stats[dataset_repo] = {
            "episodes_checked": ds_totals["episodes_checked"],
            "errors": ds_totals["errors"],
            "episodes_flagged": ds_totals["episodes_flagged"],
            "tests": {
                "duplicate_pairs": dict(ds_tests["duplicate_pairs"]),
                "scale_consistency": dict(ds_tests["scale_consistency"]),
                "tracking_residual": dict(ds_tests["tracking_residual"]),
                "delta_detection": {
                    "flagged_episodes": ds_tests["delta_detection"]["flagged_episodes"],
                    "flagged_keys": dict(delta_flagged_keys),
                },
                "length_mismatch": dict(ds_tests["length_mismatch"]),
            },
        }
        totals["episodes_checked"] += ds_totals["episodes_checked"]
        totals["episodes_skipped"] += ds_totals["errors"]
        totals["episodes_flagged"] += ds_totals["episodes_flagged"]

    report = {
        "schema_version": 1,
        "generated_at": now_timestamps(),
        "scope": f"local:{data_dir}",
        "config_snapshot": qa_config,
        "totals": dict(totals),
        "per_dataset": per_dataset_stats,
        "flagged_episodes": flagged_episodes[:MAX_FLAGGED_DETAIL],
    }
    write_stats_json(args.output, report)

    flagged_datasets = sorted(ds for ds, s in per_dataset_stats.items() if s["episodes_flagged"] > 0)
    _log("")
    _log(
        f"Checked {totals['episodes_checked']} episode(s) across {len(per_dataset_stats)} "
        f"dataset(s): {totals['episodes_flagged']} flagged, {totals['episodes_skipped']} errored."
    )
    if flagged_datasets:
        _log(f"Flagged datasets: {', '.join(flagged_datasets)}")
    else:
        _log("No datasets flagged.")
    _log(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
