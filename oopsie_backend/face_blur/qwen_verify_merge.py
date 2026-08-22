"""Merge resumable Qwen face-verification shards."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from oopsie_backend.face_blur.hf_scan import _atomic_json
from oopsie_backend.face_blur.qwen_verify import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, required=True)
    return parser


def _load_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_videos <= 0:
        parser.error("--expected-videos must be positive")

    config: dict | None = None
    by_video: dict[str, dict] = {}
    input_dirs = [path.resolve() for path in args.input_dirs]
    try:
        for input_dir in input_dirs:
            candidate_config = json.loads(
                (input_dir / "config.json").read_text(encoding="utf-8")
            )
            if config is None:
                config = candidate_config
            elif candidate_config.get("config_hash") != config.get("config_hash"):
                raise ValueError(
                    f"configuration mismatch in {input_dir}: "
                    f"{candidate_config.get('config_hash')} != "
                    f"{config.get('config_hash')}"
                )
            for result_path in sorted((input_dir / "results").glob("video_*.json.gz")):
                result = _load_gzip_json(result_path)
                video_id = str(result.get("source", {}).get("video_id") or "")
                if not video_id:
                    raise ValueError(f"result has no source.video_id: {result_path}")
                previous = by_video.get(video_id)
                if previous is None or (
                    previous.get("status") != "ok" and result.get("status") == "ok"
                ):
                    by_video[video_id] = result
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        gzip.BadGzipFile,
    ) as exc:
        print(f"error: merge: {type(exc).__name__}: {exc}")
        return 1
    if config is None:
        print("error: merge: no input configuration")
        return 1
    if len(by_video) != args.expected_videos:
        print(
            f"error: merge: found {len(by_video)} unique videos; "
            f"expected {args.expected_videos}"
        )
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "config.json", config)
    report = write_reports(output_dir, list(by_video.values()), config)
    report["inputs"] = [str(path) for path in input_dirs]
    _atomic_json(output_dir / "scan-report.json", report)
    totals = report["totals"]
    print(
        f"merged {totals['candidate_videos']} candidate videos; "
        f"verified {totals['verified_videos']}; "
        f"rejected {totals['rejected_videos']}; "
        f"incomplete {totals['incomplete_videos']}; "
        f"failed {totals['failed_videos']}"
    )
    return 1 if totals["failed_videos"] or totals["incomplete_videos"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
