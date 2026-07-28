"""Merge resumable Hugging Face face-scan shards into final CSV labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oopsie_backend.face_blur.hf_scan import (
    SCHEMA_VERSION,
    _atomic_csv,
    _atomic_json,
    _load_json,
    aggregate_episode_rows,
    result_to_video_row,
    utc_now,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-videos",
        type=int,
        default=None,
        help="Fail unless this many unique video results are present.",
    )
    return parser


def _load_results(input_dirs: list[Path]) -> tuple[dict, list[dict]]:
    config: dict | None = None
    by_video: dict[str, dict] = {}
    for input_dir in input_dirs:
        candidate_config = _load_json(input_dir / "config.json")
        if config is None:
            config = candidate_config
        elif candidate_config.get("config_hash") != config.get("config_hash"):
            raise ValueError(
                f"configuration mismatch in {input_dir}: "
                f"{candidate_config.get('config_hash')} != {config.get('config_hash')}"
            )
        for result_path in sorted((input_dir / "results").glob("video_*.json")):
            result = _load_json(result_path)
            video_id = str(result.get("source", {}).get("video_id") or "")
            if not video_id:
                raise ValueError(f"result has no source.video_id: {result_path}")
            previous = by_video.get(video_id)
            if previous is None or (
                previous.get("status") != "ok" and result.get("status") == "ok"
            ):
                by_video[video_id] = result
    if config is None:
        raise ValueError("no input directories provided")
    return config, list(by_video.values())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dirs = [path.resolve() for path in args.input_dirs]
    try:
        config, results = _load_results(input_dirs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: merge: {type(exc).__name__}: {exc}")
        return 1
    if args.expected_videos is not None and len(results) != args.expected_videos:
        print(
            f"error: merge: found {len(results)} unique videos; "
            f"expected {args.expected_videos}"
        )
        return 1

    video_rows = sorted(
        (result_to_video_row(result) for result in results),
        key=lambda row: (
            str(row.get("source_repo")),
            str(row.get("source_path")),
        ),
    )
    episode_rows = aggregate_episode_rows(video_rows)
    inventory_rows = [
        {
            "repo_id": row.get("source_repo"),
            "repo_revision": row.get("repo_revision"),
            "source_path": row.get("source_path"),
            "episode_key": row.get("episode_key"),
            "video_id": row.get("video_id"),
        }
        for row in video_rows
    ]
    output_dir = args.output_dir.resolve()
    _atomic_json(output_dir / "config.json", config)
    _atomic_csv(output_dir / "inventory.csv", inventory_rows)
    _atomic_csv(output_dir / "videos.csv", video_rows)
    _atomic_csv(output_dir / "episodes.csv", episode_rows)
    repositories = {str(row.get("source_repo")) for row in video_rows}
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": config,
        "inputs": [str(path) for path in input_dirs],
        "totals": {
            "repositories": len(repositories),
            "videos": len(video_rows),
            "episodes": len(episode_rows),
            "videos_with_faces": sum(
                row.get("has_face") is True for row in video_rows
            ),
            "episodes_with_faces": sum(
                row.get("has_face") is True for row in episode_rows
            ),
            "failed_videos": sum(row.get("status") != "ok" for row in video_rows),
        },
        "outputs": {
            "inventory": "inventory.csv",
            "videos": "videos.csv",
            "episodes": "episodes.csv",
        },
    }
    _atomic_json(output_dir / "scan-report.json", report)
    print(
        f"merged {len(video_rows)} videos / {len(episode_rows)} episodes; "
        f"faces detected in {report['totals']['videos_with_faces']} videos / "
        f"{report['totals']['episodes_with_faces']} episodes"
    )
    return 1 if report["totals"]["failed_videos"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
