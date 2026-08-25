"""Add a stable per-video face-blur tag index to the published datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from oopsie_backend.scripts.publish_face_blur_corrections import (
    _corrected_rows,
    _load_json,
)


TAG_PATH = "video_tags.csv"
TAG_FIELDS = (
    "video_path",
    "tag",
    "release",
    "blurred_sha256",
    "config_hash",
)


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


def _tag_bytes(rows: list[dict[str, object]], release: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TAG_FIELDS)
    writer.writeheader()
    for row in sorted(rows, key=lambda item: str(item["source_path"])):
        writer.writerow(
            {
                "video_path": row["source_path"],
                "tag": "face-blurred",
                "release": release,
                "blurred_sha256": row["blurred_sha256"],
                "config_hash": row["config_hash"],
            }
        )
    return output.getvalue().encode("utf-8")


def publish(
    *,
    review_manifest: Path,
    correction_validation: Path,
    original_manifest: Path,
    main_publication_path: Path,
    tag_publication_path: Path,
    release: str,
    expected_identity: str,
) -> None:
    _, full_rows, _, _ = _corrected_rows(
        review_manifest=review_manifest,
        original_manifest=original_manifest,
        correction_validation=correction_validation,
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in full_rows:
        grouped[str(row["source_repo"])].append(row)

    main_publication = _load_json(main_publication_path)
    if not main_publication.get("all_remote_verified"):
        raise ValueError("main face-blur publication has not been remotely verified")
    if set(grouped) != set(main_publication["repositories"]):
        raise ValueError("tag repositories differ from main publication repositories")

    api = HfApi()
    identity = api.whoami().get("name")
    if identity != expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {expected_identity!r}"
        )
    publication = (
        _load_json(tag_publication_path)
        if tag_publication_path.is_file()
        else {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tag": "face-blurred",
            "tag_path": TAG_PATH,
            "video_count": len(full_rows),
            "repositories": {},
        }
    )

    for repo in sorted(grouped):
        rows = grouped[repo]
        value = _tag_bytes(rows, release)
        expected_parent = main_publication["repositories"][repo]["commit"]
        previous = publication["repositories"].get(repo)
        head = api.repo_info(repo, repo_type="dataset", revision="main").sha
        if previous:
            if head != previous["commit"]:
                raise ValueError(
                    f"tagged main branch moved for {repo}: {head} != {previous['commit']}"
                )
            commit_oid = previous["commit"]
            if previous.get("remote_verified"):
                print(f"already tagged and verified: {repo}", flush=True)
                continue
        else:
            if head != expected_parent:
                raise ValueError(
                    f"refusing to write tags to changed main branch for {repo}: "
                    f"{head} != {expected_parent}"
                )
            commit = api.create_commit(
                repo,
                repo_type="dataset",
                revision="main",
                parent_commit=expected_parent,
                operations=[
                    CommitOperationAdd(path_in_repo=TAG_PATH, path_or_fileobj=value)
                ],
                commit_message=f"Tag {len(rows)} videos as face-blurred",
                commit_description=(
                    "Adds a path-keyed metadata index for filtering privacy-processed "
                    "videos. Video bytes are unchanged by this commit."
                ),
            )
            commit_oid = commit.oid
            publication["repositories"][repo] = {
                "video_count": len(rows),
                "commit": commit_oid,
                "commit_url": commit.commit_url,
                "previous_main_commit": expected_parent,
                "remote_verified": False,
            }
            _atomic_json(tag_publication_path, publication)
            print(f"tagged {repo}: {commit_oid}", flush=True)

        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=commit_oid,
                filename=TAG_PATH,
                token=api.token,
                force_download=True,
            )
        )
        if downloaded.read_bytes() != value:
            raise ValueError(f"remote tag index differs from generated index: {repo}")
        publication["repositories"][repo]["remote_verified"] = True
        _atomic_json(tag_publication_path, publication)
        print(f"verified {len(rows)} tags for {repo}", flush=True)

    publication["completed_at"] = datetime.now(timezone.utc).isoformat()
    publication["all_remote_verified"] = all(
        item.get("remote_verified") for item in publication["repositories"].values()
    )
    _atomic_json(tag_publication_path, publication)
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--correction-validation", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--main-publication", type=Path, required=True)
    parser.add_argument("--tag-publication", type=Path, required=True)
    parser.add_argument("--release", default="2026-08-12")
    parser.add_argument("--expected-identity", default="oopsiedata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publish(
        review_manifest=args.review_manifest.resolve(),
        correction_validation=args.correction_validation.resolve(),
        original_manifest=args.original_manifest.resolve(),
        main_publication_path=args.main_publication.resolve(),
        tag_publication_path=args.tag_publication.resolve(),
        release=args.release,
        expected_identity=args.expected_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
