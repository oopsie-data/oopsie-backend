"""Freeze reviewed face-blur decisions into a checksumed release bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from oopsie_backend.face_blur.benchmark import load_manifest


SNAPSHOT_FILES = (
    "annotations.csv",
    "annotations.history.jsonl",
    "setup-report.json",
    "README.md",
)
RAW_MANIFEST_FIELDS = (
    "sample_id",
    "source_repo",
    "source_revision",
    "source_path",
    "archive_path",
    "size_bytes",
    "sha256",
    "face_present",
    "blur_action",
    "annotation_source",
    "annotated_at",
    "labeler",
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
            writer = csv.DictWriter(handle, fieldnames=RAW_MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_hub_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe Hugging Face path: {value!r}")
    return path


def _complete_annotation(sample) -> bool:
    if sample.face_present == "no":
        return sample.blur_action == "preserve" and not sample.exemption_reason
    if sample.face_present == "uncertain":
        return sample.blur_action == "review" and not sample.exemption_reason
    if sample.face_present != "yes":
        return False
    if sample.blur_action == "blur":
        return not sample.exemption_reason
    return sample.blur_action == "preserve" and sample.exemption_reason in {
        "already_blurred",
        "task_relevant_depiction",
        "other",
    }


def _read_annotation_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("annotation manifest contains duplicate sample_id values")
    return by_id


def freeze_release(
    *,
    annotation_dir: Path,
    output_dir: Path,
    archive_repo: str,
    release_tag: str,
    review_branch: str,
) -> None:
    annotation_dir = annotation_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"release directory is not empty: {output_dir}")

    annotations_path = annotation_dir / "annotations.csv"
    annotation_rows = _read_annotation_rows(annotations_path)
    samples = load_manifest(annotations_path)
    incomplete = [sample.sample_id for sample in samples if not _complete_annotation(sample)]
    if incomplete:
        preview = ", ".join(incomplete[:5])
        raise ValueError(
            f"refusing to freeze {len(incomplete)} incomplete annotation(s): {preview}"
        )

    selected = [sample for sample in samples if sample.blur_action == "blur"]
    if not selected:
        raise ValueError("manifest selects no videos for blurring")

    seen_source_keys: set[tuple[str, str]] = set()
    raw_rows: list[dict[str, object]] = []
    source_revisions: dict[str, set[str]] = defaultdict(set)
    bytes_by_repo: Counter[str] = Counter()
    count_by_repo: Counter[str] = Counter()
    for sample in selected:
        source = sample.video_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"selected source video is missing: {source}")
        row = annotation_rows[sample.sample_id]
        repo = row["source_repo"].strip()
        revision = row["repo_revision"].strip()
        source_path = _safe_hub_path(row["source_path"].strip())
        if not repo or not revision:
            raise ValueError(f"missing source repo/revision for {sample.sample_id}")
        source_key = (repo, source_path.as_posix())
        if source_key in seen_source_keys:
            raise ValueError(f"duplicate source repo/path: {source_key}")
        seen_source_keys.add(source_key)
        repo_name = _safe_hub_path(repo.rsplit("/", 1)[-1])
        archive_path = PurePosixPath("raw", repo_name, source_path)
        size_bytes = source.stat().st_size
        raw_rows.append(
            {
                "sample_id": sample.sample_id,
                "source_repo": repo,
                "source_revision": revision,
                "source_path": source_path.as_posix(),
                "archive_path": archive_path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": _sha256(source),
                "face_present": sample.face_present,
                "blur_action": sample.blur_action,
                "annotation_source": row["annotation_source"],
                "annotated_at": row["annotated_at"],
                "labeler": row["labeler"],
            }
        )
        source_revisions[repo].add(revision)
        bytes_by_repo[repo] += size_bytes
        count_by_repo[repo] += 1

    multiple_revisions = {
        repo: sorted(revisions)
        for repo, revisions in source_revisions.items()
        if len(revisions) != 1
    }
    if multiple_revisions:
        raise ValueError(f"repositories span multiple revisions: {multiple_revisions}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES:
        source = annotation_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    _atomic_csv(output_dir / "raw_manifest.csv", raw_rows)

    generated_at = datetime.now(timezone.utc).isoformat()
    release = {
        "schema_version": 1,
        "generated_at": generated_at,
        "archive_repo": archive_repo,
        "release_tag": release_tag,
        "review_branch": review_branch,
        "source_manifest": str(annotations_path),
        "source_manifest_sha256": _sha256(annotations_path),
        "annotation_count": len(samples),
        "blur_video_count": len(raw_rows),
        "blur_video_bytes": sum(int(row["size_bytes"]) for row in raw_rows),
        "source_revisions": {
            repo: next(iter(revisions))
            for repo, revisions in sorted(source_revisions.items())
        },
        "blur_counts_by_repo": dict(sorted(count_by_repo.items())),
        "blur_bytes_by_repo": dict(sorted(bytes_by_repo.items())),
    }
    _atomic_json(output_dir / "release.json", release)

    checksum_paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_text = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in checksum_paths
    )
    (output_dir / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")
    print(json.dumps(release, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-repo", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--review-branch", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_release(
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        archive_repo=args.archive_repo,
        release_tag=args.release_tag,
        review_branch=args.review_branch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
