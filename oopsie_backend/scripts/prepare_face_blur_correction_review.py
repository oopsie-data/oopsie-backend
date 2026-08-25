"""Validate fallback corrections and prepare a manual review manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from oopsie_backend.face_blur.cli import discover_jobs
from oopsie_backend.face_blur.pipeline import VideoInfo, validate_encoded_video
from oopsie_backend.face_blur.tracked_deface_cli import metadata_path_for
from oopsie_backend.scripts.publish_face_blur_release import _validate_full_decode


REVIEW_FIELDS = (
    "sample_id",
    "source_repo",
    "source_path",
    "raw_video_path",
    "previous_video_path",
    "corrected_video_path",
    "suspected_reason",
    "old_masked_frame_count",
    "corrected_masked_frame_count",
    "old_region_count",
    "corrected_region_count",
    "fallback_observation_count",
    "fallback_temporal_region_count",
    "review_decision",
    "review_notes",
    "reviewed_at",
    "reviewer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_review(
    *,
    correction_manifest: Path,
    previous_manifest: Path,
    output_dir: Path,
    review_dir: Path,
) -> dict:
    previous_rows = _load_csv(previous_manifest)
    previous_by_id = {row["sample_id"]: row for row in previous_rows}
    jobs = discover_jobs(
        output_dir=output_dir.resolve(),
        manifest=correction_manifest.resolve(),
        blur_actions={"blur"},
        require_complete_actions=True,
    )
    if len(jobs) != 97 or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError(f"expected 97 unique correction jobs, found {len(jobs)}")
    if any(job.job_id not in previous_by_id for job in jobs):
        raise ValueError("correction set contains a video absent from the prior release")

    run_path = output_dir / "tracked-deface-run.json"
    run = _load_json(run_path)
    totals = run.get("totals", {})
    if totals != {
        **totals,
        "videos": 97,
        "successful": 97,
        "failed": 0,
    }:
        if (
            totals.get("videos") != 97
            or totals.get("successful") != 97
            or totals.get("failed") != 0
        ):
            raise ValueError(f"correction generation is incomplete: {totals}")
    config = run.get("config", {})
    confirmation = config.get("confirmation", {})
    if config.get("algorithm_version") != 3:
        raise ValueError(f"unexpected correction algorithm version: {config}")
    if (
        confirmation.get("fallback_pre_roll_frames") != 2
        or confirmation.get("fallback_hold_frames") != 15
    ):
        raise ValueError(f"unexpected fallback policy: {confirmation}")

    review_rows: list[dict[str, object]] = []
    errors: list[str] = []
    for index, job in enumerate(jobs, 1):
        old = previous_by_id[job.job_id]
        try:
            corrected = job.output_path
            sidecar = _load_json(metadata_path_for(corrected))
            if sidecar.get("status") != "ok":
                raise ValueError(f"sidecar status is {sidecar.get('status')!r}")
            if sidecar.get("source", {}).get("sha256") != old["source_sha256"]:
                raise ValueError("raw source hash differs from prior release")
            if sidecar.get("output", {}).get("sha256") != _sha256(corrected):
                raise ValueError("corrected output hash differs from sidecar")
            video = sidecar["video"]
            video_info = VideoInfo(
                width=int(video["width"]),
                height=int(video["height"]),
                fps=float(video["fps"]),
                expected_frame_count=int(video["expected_frame_count"]),
                decoded_frame_count=int(video["decoded_frame_count"]),
            )
            validate_encoded_video(corrected, video_info)
            _validate_full_decode(corrected, video_info)
            tracking = sidecar.get("tracking", {})
            policy = tracking.get("policy", {})
            if int(policy.get("fallback_track_count") or 0) <= 0:
                raise ValueError("corrected output no longer reports a fallback track")
            temporal_regions = sum(
                int(count)
                for source, count in tracking.get("source_counts", {}).items()
                if source in {
                    "fallback_pre_roll",
                    "fallback_interpolated",
                    "fallback_hold",
                }
            )
            if temporal_regions <= 0:
                raise ValueError("fallback received no temporal coverage")
            corrected_masked = int(tracking.get("masked_frame_count") or 0)
            old_masked = int(old["masked_frame_count"])
            if corrected_masked < old_masked:
                raise ValueError(
                    f"masked-frame coverage decreased: {corrected_masked} < {old_masked}"
                )
            reason = ["fallback_detection"]
            if int(old["confirmed_track_count"]) == 0:
                reason.append("fallback_only")
            if old_masked <= 8:
                reason.append("very_low_prior_coverage")
            if job.job_id in {
                "video_424c6314cbbc61a6",
                "video_68466cb6cce96b31",
                "video_df39dcfcf6e1ffa5",
                "video_7403d0eecd45c241",
            }:
                reason.append("visually_confirmed_prior_miss")
            review_rows.append(
                {
                    "sample_id": job.job_id,
                    "source_repo": old["source_repo"],
                    "source_path": old["source_path"],
                    "raw_video_path": str(job.source_path.resolve()),
                    "previous_video_path": old["output_local_path"],
                    "corrected_video_path": str(corrected.resolve()),
                    "suspected_reason": "|".join(reason),
                    "old_masked_frame_count": old_masked,
                    "corrected_masked_frame_count": corrected_masked,
                    "old_region_count": old["region_count"],
                    "corrected_region_count": int(tracking.get("region_count") or 0),
                    "fallback_observation_count": int(
                        policy.get("fallback_observation_count") or 0
                    ),
                    "fallback_temporal_region_count": temporal_regions,
                    "review_decision": "",
                    "review_notes": "",
                    "reviewed_at": "",
                    "reviewer": "",
                }
            )
        except Exception as exc:  # collect every failure
            errors.append(f"{job.job_id}: {type(exc).__name__}: {exc}")
        if index % 20 == 0 or index == len(jobs):
            print(f"validated {index}/{len(jobs)} corrections", flush=True)

    report = {
        "schema_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "valid": not errors and len(review_rows) == 97,
        "video_count": len(jobs),
        "valid_video_count": len(review_rows),
        "errors": errors,
        "correction_manifest": str(correction_manifest.resolve()),
        "correction_manifest_sha256": _sha256(correction_manifest),
        "previous_manifest": str(previous_manifest.resolve()),
        "previous_manifest_sha256": _sha256(previous_manifest),
        "run_report": str(run_path.resolve()),
        "run_report_sha256": _sha256(run_path),
        "config": config,
        "model": run.get("model"),
        "runtime": run.get("runtime"),
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(review_dir / "validation.json", report)
    if errors or len(review_rows) != 97:
        raise ValueError(
            f"correction validation failed; see {review_dir / 'validation.json'}"
        )
    _atomic_csv(review_dir / "review.csv", review_rows)
    (review_dir / "review.history.jsonl").touch(exist_ok=True)
    print(f"manual review manifest: {review_dir / 'review.csv'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-manifest", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_review(
        correction_manifest=args.correction_manifest,
        previous_manifest=args.previous_manifest,
        output_dir=args.output_dir,
        review_dir=args.review_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
