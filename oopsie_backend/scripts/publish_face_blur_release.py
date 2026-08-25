"""Validate blurred videos and publish them to pinned Hugging Face review branches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from oopsie_backend.face_blur.cli import discover_jobs
from oopsie_backend.face_blur.pipeline import VideoInfo, validate_encoded_video
from oopsie_backend.face_blur.tracked_deface_cli import metadata_path_for


GENERATED_FIELDS = (
    "sample_id",
    "source_repo",
    "source_revision",
    "source_path",
    "archive_path",
    "source_size_bytes",
    "source_sha256",
    "output_local_path",
    "blurred_size_bytes",
    "blurred_sha256",
    "masked_frame_count",
    "region_count",
    "track_count",
    "confirmed_track_count",
    "fallback_detection_count",
    "config_hash",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_full_decode(path: Path, expected: VideoInfo) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open encoded output: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != expected.width or frame.shape[0] != expected.height:
                raise RuntimeError(
                    f"dimensions changed during decode at frame {frame_count}: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            frame_count += 1
    finally:
        capture.release()
    if (width, height) != (expected.width, expected.height):
        raise RuntimeError(
            f"encoded dimensions differ: {width}x{height} != "
            f"{expected.width}x{expected.height}"
        )
    if abs(fps - expected.fps) > max(0.01, expected.fps * 0.001):
        raise RuntimeError(f"encoded FPS differs: {fps} != {expected.fps}")
    if frame_count != expected.decoded_frame_count:
        raise RuntimeError(
            f"full decode frame count differs: {frame_count} != "
            f"{expected.decoded_frame_count}"
        )


def validate_release(
    *, manifest: Path, release_dir: Path, output_dir: Path
) -> tuple[list[dict[str, object]], dict]:
    release = _load_json(release_dir / "release.json")
    actual_manifest_sha = _sha256(manifest)
    if actual_manifest_sha != release["source_manifest_sha256"]:
        raise ValueError(
            "annotation manifest checksum differs from the frozen release: "
            f"{actual_manifest_sha} != {release['source_manifest_sha256']}"
        )

    raw_rows = _load_csv(release_dir / "raw_manifest.csv")
    raw_by_id = {row["sample_id"]: row for row in raw_rows}
    jobs = discover_jobs(
        output_dir=output_dir.resolve(),
        manifest=manifest,
        blur_actions={"blur"},
        require_complete_actions=True,
    )
    jobs_by_id = {job.job_id: job for job in jobs}
    if len(raw_by_id) != len(raw_rows) or len(jobs_by_id) != len(jobs):
        raise ValueError("duplicate sample_id in raw manifest or blur jobs")
    if set(raw_by_id) != set(jobs_by_id):
        missing = sorted(set(raw_by_id) ^ set(jobs_by_id))
        raise ValueError(f"raw manifest and blur jobs differ: {missing[:10]}")
    if len(jobs) != int(release["blur_video_count"]):
        raise ValueError(
            f"expected {release['blur_video_count']} blur jobs, found {len(jobs)}"
        )

    run_report_path = output_dir / "tracked-deface-run.json"
    run_report = _load_json(run_report_path)
    totals = run_report.get("totals", {})
    if (
        totals.get("videos") != len(jobs)
        or totals.get("successful") != len(jobs)
        or totals.get("failed") != 0
    ):
        raise ValueError(f"tracked Deface run is incomplete: {totals}")
    result_ids = {row.get("job_id") for row in run_report.get("results", [])}
    if result_ids != set(jobs_by_id):
        raise ValueError("tracked Deface run report does not cover the exact release set")

    errors: list[str] = []
    generated_rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sorted(jobs_by_id), 1):
        job = jobs_by_id[sample_id]
        raw = raw_by_id[sample_id]
        output = job.output_path
        sidecar_path = metadata_path_for(output)
        try:
            if not output.is_file():
                raise FileNotFoundError(f"output missing: {output}")
            sidecar = _load_json(sidecar_path)
            if sidecar.get("status") != "ok":
                raise ValueError(f"sidecar status is not ok: {sidecar.get('status')}")
            source = sidecar.get("source", {})
            encoded = sidecar.get("output", {})
            tracking = sidecar.get("tracking", {})
            policy = tracking.get("policy", {})
            source_sha = _sha256(job.source_path)
            output_sha = _sha256(output)
            if source_sha != raw["sha256"] or source.get("sha256") != raw["sha256"]:
                raise ValueError("source SHA-256 does not match frozen raw manifest")
            if job.source_path.stat().st_size != int(raw["size_bytes"]):
                raise ValueError("source byte count does not match frozen raw manifest")
            if encoded.get("sha256") != output_sha:
                raise ValueError("output SHA-256 does not match audit sidecar")
            if encoded.get("size_bytes") != output.stat().st_size:
                raise ValueError("output byte count does not match audit sidecar")
            if output_sha == source_sha:
                raise ValueError("blurred file is byte-identical to its source")
            masked_frames = int(tracking.get("masked_frame_count") or 0)
            if masked_frames <= 0:
                raise ValueError("selected face video has zero masked frames")
            video = sidecar.get("video", {})
            video_info = VideoInfo(
                width=int(video["width"]),
                height=int(video["height"]),
                fps=float(video["fps"]),
                expected_frame_count=int(video["expected_frame_count"]),
                decoded_frame_count=int(video["decoded_frame_count"]),
            )
            validate_encoded_video(output, video_info)
            _validate_full_decode(output, video_info)
            generated_rows.append(
                {
                    "sample_id": sample_id,
                    "source_repo": raw["source_repo"],
                    "source_revision": raw["source_revision"],
                    "source_path": raw["source_path"],
                    "archive_path": raw["archive_path"],
                    "source_size_bytes": raw["size_bytes"],
                    "source_sha256": raw["sha256"],
                    "output_local_path": str(output.resolve()),
                    "blurred_size_bytes": output.stat().st_size,
                    "blurred_sha256": output_sha,
                    "masked_frame_count": masked_frames,
                    "region_count": int(tracking.get("region_count") or 0),
                    "track_count": int(tracking.get("track_count") or 0),
                    "confirmed_track_count": int(
                        policy.get("confirmed_track_count") or 0
                    ),
                    "fallback_detection_count": int(
                        policy.get("fallback_detection_count") or 0
                    ),
                    "config_hash": sidecar["config_hash"],
                }
            )
        except Exception as exc:  # continue to report every invalid output
            errors.append(f"{sample_id}: {type(exc).__name__}: {exc}")
        if index % 25 == 0 or index == len(jobs):
            print(f"validated {index}/{len(jobs)} local outputs", flush=True)

    generated_dir = release_dir / "generated"
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "valid": not errors and len(generated_rows) == len(jobs),
        "expected_videos": len(jobs),
        "valid_videos": len(generated_rows),
        "errors": errors,
        "run_report": str(run_report_path),
        "run_report_sha256": _sha256(run_report_path),
        "config": run_report.get("config"),
        "model": run_report.get("model"),
        "runtime": run_report.get("runtime"),
    }
    _atomic_json(generated_dir / "validation.json", validation)
    if errors or len(generated_rows) != len(jobs):
        raise ValueError(
            f"release validation failed for {len(errors)} of {len(jobs)} videos; "
            f"see {generated_dir / 'validation.json'}"
        )
    _atomic_csv(generated_dir / "blurred_manifest.csv", generated_rows)
    return generated_rows, validation


def _repo_manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=GENERATED_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def publish_release(
    *,
    rows: list[dict[str, object]],
    validation: dict,
    release_dir: Path,
    expected_identity: str,
    verify_remote: bool,
) -> dict:
    release = _load_json(release_dir / "release.json")
    api = HfApi()
    identity = api.whoami().get("name")
    if identity != expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {expected_identity!r}"
        )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_repo"])].append(row)
    if set(grouped) != set(release["source_revisions"]):
        raise ValueError("generated output repositories differ from frozen release")

    generated_dir = release_dir / "generated"
    publication_path = generated_dir / "publication.json"
    publication = (
        _load_json(publication_path)
        if publication_path.is_file()
        else {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "review_branch": release["review_branch"],
            "repositories": {},
        }
    )
    for repo in sorted(grouped):
        repo_rows = sorted(grouped[repo], key=lambda row: str(row["source_path"]))
        expected_revision = release["source_revisions"][repo]
        previous = publication["repositories"].get(repo, {})
        if previous.get("remote_verified"):
            print(f"already published and verified: {repo}", flush=True)
            continue
        branch_head = api.repo_info(
            repo, repo_type="dataset", revision=release["review_branch"]
        ).sha
        if branch_head != expected_revision:
            raise ValueError(
                f"refusing to overwrite changed review branch for {repo}: "
                f"{branch_head} != {expected_revision}"
            )
        operations = [
            CommitOperationAdd(
                path_in_repo=str(row["source_path"]),
                path_or_fileobj=Path(str(row["output_local_path"])),
            )
            for row in repo_rows
        ]
        metadata_root = "face_blur_release/2026-08-12"
        operations.extend(
            [
                CommitOperationAdd(
                    path_in_repo=f"{metadata_root}/manifest.csv",
                    path_or_fileobj=_repo_manifest_bytes(repo_rows),
                ),
                CommitOperationAdd(
                    path_in_repo=f"{metadata_root}/validation.json",
                    path_or_fileobj=(
                        json.dumps(validation, indent=2, sort_keys=True) + "\n"
                    ).encode(),
                ),
                CommitOperationAdd(
                    path_in_repo=f"{metadata_root}/README.md",
                    path_or_fileobj=(
                        "# Face-blur review release (2026-08-12)\n\n"
                        f"This branch replaces {len(repo_rows)} manually verified face-containing "
                        "videos with tracked Deface outputs. It does not modify `main`. "
                        "Raw originals are recoverable from the private archive and from the "
                        "`pre-face-blur-2026-08-12` tag. See `manifest.csv` for hashes and "
                        "provenance.\n"
                    ).encode(),
                ),
            ]
        )
        commit = api.create_commit(
            repo,
            repo_type="dataset",
            revision=release["review_branch"],
            parent_commit=expected_revision,
            operations=operations,
            commit_message=f"Add {len(repo_rows)} reviewed face-blurred videos",
            commit_description=(
                "Staged privacy release; raw originals are preserved in the private "
                "archive and by the pre-face-blur tag."
            ),
            num_threads=8,
        )
        publication["repositories"][repo] = {
            "video_count": len(repo_rows),
            "commit": commit.oid,
            "commit_url": commit.commit_url,
            "remote_verified": False,
        }
        _atomic_json(publication_path, publication)
        print(f"published {repo}: {commit.oid}", flush=True)

        if verify_remote:
            for index, row in enumerate(repo_rows, 1):
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo,
                        repo_type="dataset",
                        revision=commit.oid,
                        filename=str(row["source_path"]),
                    )
                )
                if (
                    downloaded.stat().st_size != int(row["blurred_size_bytes"])
                    or _sha256(downloaded) != row["blurred_sha256"]
                ):
                    raise ValueError(
                        f"remote hash verification failed: {repo}/{row['source_path']}"
                    )
                if index % 25 == 0 or index == len(repo_rows):
                    print(
                        f"verified {index}/{len(repo_rows)} remote outputs for {repo}",
                        flush=True,
                    )
            publication["repositories"][repo]["remote_verified"] = True
            _atomic_json(publication_path, publication)

    publication["completed_at"] = datetime.now(timezone.utc).isoformat()
    publication["all_remote_verified"] = all(
        item.get("remote_verified") for item in publication["repositories"].values()
    )
    _atomic_json(publication_path, publication)
    return publication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument("--expected-identity", default="oopsiedata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, validation = validate_release(
        manifest=args.manifest.resolve(),
        release_dir=args.release_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"local release valid: {len(rows)} videos", flush=True)
    if args.publish:
        publication = publish_release(
            rows=rows,
            validation=validation,
            release_dir=args.release_dir.resolve(),
            expected_identity=args.expected_identity,
            verify_remote=args.verify_remote,
        )
        print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
