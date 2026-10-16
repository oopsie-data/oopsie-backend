"""Run the action-space data quality checks (oopsie_backend.utils.action_qa) against
real per-timestep actions/* and observations/robot_states/* arrays, and the video checks
(oopsie_backend.utils.video_checks) against each episode's camera videos.

This is a separate, opt-in pass from compute_local_stats.py: it reads full arrays
(not just HDF5 attrs), which costs much more per file, so it is not run as part of
the nightly stats job. See reports/data_quality_exceptions.md for what each check
means and configs/action_qa_config.yaml for toggles/thresholds.

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
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py

from oopsie_backend.scripts.compute_local_stats import DEFAULT_DATA_DIR, discover_episodes
from oopsie_backend.utils.action_qa import (
    load_action_state_arrays,
    profile_control_freq,
    run_action_qa_on_arrays,
    run_metadata_checks,
)
from oopsie_backend.utils.action_qa_config import CONFIG as ACTION_QA_CONFIG, DEFAULT_CHECKS
from oopsie_backend.utils.action_qa_report import render_action_qa_report
from oopsie_backend.utils.episode_stats import (
    PUBLIC_EXCLUDED_DATASETS,
    STATS_EXCLUDED_DATASETS,
    now_timestamps,
    write_stats_json,
)
from oopsie_backend.utils.qa_sampling import episode_videos
from oopsie_backend.utils.video_checks import run_video_checks

MAX_FLAGGED_DETAIL = 20000


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _merge(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    return {key: a[key] + b[key] for key in ("findings", "skips", "checked")}


def _check_episode(job: tuple[Path, str, bool], qa_config: dict) -> tuple[dict | None, str | None]:
    """Action and video QA for one episode; the video checks only when ``job``'s flag
    says so (see sampling.video_episode_limit). Returns ``(result, error)``; ``result``
    is None when nothing applies (no declared action space and no video checks).
    """
    path, repo_path, run_video = job
    try:
        with h5py.File(path, "r") as f:
            actions, states, profile = load_action_state_arrays(f)
            metadata = run_metadata_checks(f, profile, qa_config)
        result = run_action_qa_on_arrays(actions, states, profile, qa_config)
        if metadata["checked"]:
            result = _merge(result, metadata)
        video_cfg = qa_config["video"]
        if run_video and any(cfg["enabled"] for cfg in video_cfg.values()):
            videos = episode_videos(path, repo_path)
            result = _merge(
                result, run_video_checks(
                    videos, states, profile_control_freq(profile), video_cfg,
                    qa_config["video_settings"],
                )
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return result, None


def main(argv: list[str] | None = None) -> int:
    qa_config = ACTION_QA_CONFIG
    run_cfg = qa_config["run"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Default: run.data_dir from action_qa_config.yaml, else compute_local_stats.py's default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max episodes checked per dataset, first N (default: "
            "sampling.default_episode_limit from action_qa_config.yaml). "
            "0 means unlimited."
        ),
    )
    parser.add_argument(
        "--video-limit",
        type=int,
        default=None,
        help=(
            "Of each dataset's checked episodes, how many also get the video checks (default: "
            "sampling.video_episode_limit from action_qa_config.yaml). 0 means all of them."
        ),
    )
    parser.add_argument(
        "--max-total-episodes",
        type=int,
        default=None,
        help="Global cap across all datasets combined (default: run.max_total_episodes from action_qa_config.yaml).",
    )
    parser.add_argument(
        "--only-dataset",
        action="append",
        default=None,
        metavar="REPO",
        help=(
            "Restrict to this dataset repo (e.g. OopsieData-Submissions/SomeLab). Repeatable. "
            "Default: run.only_dataset from action_qa_config.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: run.output from action_qa_config.yaml.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel episodes (default: run.workers from action_qa_config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: run.verbose from action_qa_config.yaml.",
    )
    args = parser.parse_args(argv)

    data_dir_arg = args.data_dir if args.data_dir is not None else run_cfg["data_dir"]
    data_dir_arg = Path(data_dir_arg) if data_dir_arg is not None else DEFAULT_DATA_DIR
    max_total_episodes = args.max_total_episodes if args.max_total_episodes is not None else run_cfg["max_total_episodes"]
    only_dataset_list = args.only_dataset if args.only_dataset is not None else run_cfg["only_dataset"]
    output_path = args.output if args.output is not None else run_cfg["output"]
    verbose = args.verbose if args.verbose is not None else run_cfg["verbose"]
    workers = args.workers if args.workers is not None else run_cfg["workers"]

    per_dataset_limit = args.limit if args.limit is not None else qa_config["sampling"]["default_episode_limit"]
    per_dataset_limit = None if per_dataset_limit in (0, None) else per_dataset_limit
    video_limit = args.video_limit if args.video_limit is not None else qa_config["sampling"]["video_episode_limit"]
    video_limit = None if video_limit in (0, None) else video_limit
    only_dataset = set(only_dataset_list) if only_dataset_list else None
    scope_cfg = qa_config["scope"]

    data_dir = data_dir_arg.resolve()
    if not data_dir.is_dir():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    episodes = discover_episodes(data_dir)
    by_dataset: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for dataset_repo, repo_path, path in episodes:
        if dataset_repo in STATS_EXCLUDED_DATASETS and not scope_cfg["include_stats_excluded_datasets"]:
            continue
        if dataset_repo in PUBLIC_EXCLUDED_DATASETS and not scope_cfg["include_public_excluded_datasets"]:
            continue
        if only_dataset is not None and dataset_repo not in only_dataset:
            continue
        by_dataset[dataset_repo].append((repo_path, path))

    if not by_dataset:
        print("error: no matching .h5 files found", file=sys.stderr)
        return 1

    sampling_cfg = qa_config["sampling"]
    rng = random.Random(sampling_cfg["random_seed"]) if sampling_cfg["random_seed"] is not None else random.Random()

    def _select(files: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
        if per_dataset_limit is None:
            return files
        if sampling_cfg["random"]:
            return rng.sample(files, k=min(per_dataset_limit, len(files)))
        return files[:per_dataset_limit]

    planned = {ds: _select(files) for ds, files in by_dataset.items()}
    total_planned = sum(len(files) for files in planned.values())
    if max_total_episodes is not None:
        total_planned = min(total_planned, max_total_episodes)
    sample_kind = "random" if sampling_cfg["random"] else "first-N sorted"
    _log(f"Will check up to {total_planned} episode(s) across {len(planned)} dataset(s) ({sample_kind} sample)")
    _log(f"Video checks on {'all of them' if video_limit is None else f'up to {video_limit} per dataset'}")

    check_categories = {
        category: [test for test, cfg in qa_config[category].items() if cfg["enabled"]]
        for category in DEFAULT_CHECKS
    }
    totals = Counter()
    per_dataset_stats: dict[str, dict] = {}
    flagged_episodes: list[dict] = []
    checked_total = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, dataset_repo in enumerate(sorted(planned), 1):
            if max_total_episodes is not None and checked_total >= max_total_episodes:
                break
            files = planned[dataset_repo]
            if max_total_episodes is not None:
                remaining = max_total_episodes - checked_total
                files = files[:remaining]

            ds_totals = Counter()
            # One entry per enabled check, so a check that never fired still shows up as zero.
            ds_checked_episodes = {cat: {t: 0 for t in tests} for cat, tests in check_categories.items()}
            ds_flagged_episodes = {cat: {t: 0 for t in tests} for cat, tests in check_categories.items()}
            ds_flagged_keys = {cat: {t: Counter() for t in tests} for cat, tests in check_categories.items()}
            ds_skips = {cat: {t: Counter() for t in tests} for cat, tests in check_categories.items()}

            _log(f"[{index}/{len(planned)}] {dataset_repo}: {len(files)} episode(s)")
            # The sample is already in random order (or sorted, if sampling.random is off),
            # so its first video_limit episodes are a fair subset for the video checks.
            jobs = [
                (path, repo_path, video_limit is None or i < video_limit)
                for i, (repo_path, path) in enumerate(files)
            ]
            results = pool.map(partial(_check_episode, qa_config=qa_config), jobs, chunksize=2)
            for i, ((repo_path, path), (result, error)) in enumerate(zip(files, results), 1):
                checked_total += 1
                ds_totals["episodes_checked"] += 1
                if error is not None:
                    ds_totals["errors"] += 1
                    if verbose:
                        _log(f"  [{i}/{len(files)}] {repo_path}: ERROR ({error})")
                    continue

                if result is None:
                    continue
                for category, test in result["checked"]:
                    ds_checked_episodes[category][test] += 1
                for skip in result["skips"]:
                    ds_skips[skip["category"]][skip["test"]][skip["reason"]] += 1
                if not result["findings"]:
                    continue

                tests_hit = {(finding["category"], finding["test"]) for finding in result["findings"]}
                for category, test in tests_hit:
                    ds_flagged_episodes[category][test] += 1
                for finding in result["findings"]:
                    ds_flagged_keys[finding["category"]][finding["test"]][finding["key"]] += 1

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
                if verbose:
                    _log(f"  [{i}/{len(files)}] {repo_path}: FLAGGED ({sorted(f'{c}.{t}' for c, t in tests_hit)})")

            per_dataset_stats[dataset_repo] = {
                "episodes_checked": ds_totals["episodes_checked"],
                "errors": ds_totals["errors"],
                "episodes_flagged": ds_totals["episodes_flagged"],
                "tests": {
                    category: {
                        test: {
                            "checked_episodes": ds_checked_episodes[category][test],
                            "flagged_episodes": ds_flagged_episodes[category][test],
                            "flagged_keys": dict(ds_flagged_keys[category][test]),
                            "skips": dict(ds_skips[category][test]),
                        }
                        for test in tests
                    }
                    for category, tests in check_categories.items()
                },
            }
            totals["episodes_checked"] += ds_totals["episodes_checked"]
            totals["episodes_skipped"] += ds_totals["errors"]
            totals["episodes_flagged"] += ds_totals["episodes_flagged"]

    report = {
        "schema_version": 2,
        "generated_at": now_timestamps(),
        "scope": f"local:{data_dir}",
        "config_snapshot": {
            **{k: v for k, v in qa_config.items() if k != "run"},
            # The limits actually used, which --limit / --video-limit may have overridden.
            "sampling": {
                **qa_config["sampling"],
                "default_episode_limit": per_dataset_limit or 0,
                "video_episode_limit": video_limit or 0,
            },
            "run": {**qa_config["run"], "data_dir": str(data_dir), "output": str(output_path)},
        },
        "totals": dict(totals),
        "per_dataset": per_dataset_stats,
        "flagged_episodes": flagged_episodes[:MAX_FLAGGED_DETAIL],
    }
    write_stats_json(output_path, report)
    html_path = Path(output_path).with_suffix(".html")
    render_action_qa_report(report, html_path, qa_config["report"]["highlight_ignore"])

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
    _log(f"Report: {output_path}")
    _log(f"HTML:   {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
