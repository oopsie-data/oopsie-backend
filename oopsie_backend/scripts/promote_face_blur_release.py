"""Promote verified face-blur review branches to dataset main branches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from oopsie_backend.scripts.publish_face_blur_corrections import (
    _atomic_json,
    _corrected_rows,
    _load_json,
    _sha256,
)


METADATA_PATHS = (
    "face_blur_release/2026-08-12/manifest.csv",
    "face_blur_release/2026-08-12/validation.json",
    "face_blur_release/2026-08-12/correction-validation.json",
    "face_blur_release/2026-08-12/correction-review-summary.json",
    "face_blur_release/2026-08-12/README.md",
)


def _validate_local_outputs(rows: list[dict[str, object]]) -> None:
    for index, row in enumerate(rows, 1):
        output = Path(str(row["output_local_path"]))
        if not output.is_file():
            raise FileNotFoundError(f"missing final blurred output: {output}")
        if (
            output.stat().st_size != int(row["blurred_size_bytes"])
            or _sha256(output) != row["blurred_sha256"]
        ):
            raise ValueError(f"local blurred output changed: {row['sample_id']}")
        if index % 50 == 0 or index == len(rows):
            print(f"validated {index}/{len(rows)} final local files", flush=True)


def _verify_remote(
    api: HfApi,
    *,
    repo: str,
    revision: str,
    rows: list[dict[str, object]],
) -> None:
    for index, row in enumerate(rows, 1):
        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                filename=str(row["source_path"]),
                token=api.token,
            )
        )
        if (
            downloaded.stat().st_size != int(row["blurred_size_bytes"])
            or _sha256(downloaded) != row["blurred_sha256"]
        ):
            raise ValueError(f"main hash verification failed: {repo}/{row['source_path']}")
        if index % 25 == 0 or index == len(rows):
            print(f"verified {index}/{len(rows)} main files for {repo}", flush=True)


def promote(
    *,
    review_manifest: Path,
    correction_validation: Path,
    original_manifest: Path,
    release_path: Path,
    original_publication_path: Path,
    correction_publication_path: Path,
    main_publication_path: Path,
    review_branch: str,
    expected_identity: str,
) -> None:
    _, full_rows, _, _ = _corrected_rows(
        review_manifest=review_manifest,
        original_manifest=original_manifest,
        correction_validation=correction_validation,
    )
    _validate_local_outputs(full_rows)

    release = _load_json(release_path)
    original_publication = _load_json(original_publication_path)
    correction_publication = _load_json(correction_publication_path)
    if not original_publication.get("all_remote_verified"):
        raise ValueError("original review publication was not remotely verified")
    if not correction_publication.get("all_remote_verified"):
        raise ValueError("correction publication was not remotely verified")
    if release.get("review_branch") != review_branch:
        raise ValueError("review branch differs from frozen release")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in full_rows:
        grouped[str(row["source_repo"])].append(row)
    if set(grouped) != set(correction_publication["repositories"]):
        raise ValueError("final release repositories differ from correction publication")

    api = HfApi()
    identity = api.whoami().get("name")
    if identity != expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {expected_identity!r}"
        )
    publication = (
        _load_json(main_publication_path)
        if main_publication_path.is_file()
        else {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_review_branch": review_branch,
            "video_count": len(full_rows),
            "repositories": {},
        }
    )

    for repo in sorted(grouped):
        rows = sorted(grouped[repo], key=lambda row: str(row["source_path"]))
        expected_base = release["source_revisions"][repo]
        expected_review = correction_publication["repositories"][repo]["commit"]
        review_head = api.repo_info(
            repo, repo_type="dataset", revision=review_branch
        ).sha
        if review_head != expected_review:
            raise ValueError(
                f"verified review branch moved for {repo}: {review_head} != {expected_review}"
            )

        previous = publication["repositories"].get(repo)
        main_head = api.repo_info(repo, repo_type="dataset", revision="main").sha
        if previous:
            if main_head != previous["commit"]:
                raise ValueError(
                    f"published main branch moved for {repo}: "
                    f"{main_head} != {previous['commit']}"
                )
            commit_oid = previous["commit"]
            if previous.get("remote_verified"):
                print(f"already promoted and verified: {repo}", flush=True)
                continue
        else:
            if main_head != expected_base:
                raise ValueError(
                    f"refusing to overwrite changed main branch for {repo}: "
                    f"{main_head} != {expected_base}"
                )
            operations = [
                CommitOperationAdd(
                    path_in_repo=str(row["source_path"]),
                    path_or_fileobj=Path(str(row["output_local_path"])),
                )
                for row in rows
            ]
            for path_in_repo in METADATA_PATHS:
                downloaded = hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    revision=expected_review,
                    filename=path_in_repo,
                    token=api.token,
                )
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=Path(downloaded),
                    )
                )
            commit = api.create_commit(
                repo,
                repo_type="dataset",
                revision="main",
                parent_commit=expected_base,
                operations=operations,
                commit_message=f"Publish {len(rows)} reviewed face-blurred videos",
                commit_description=(
                    f"Promoted from the verified {review_branch} release. Raw originals "
                    "remain preserved in the private archive and pre-face-blur tag."
                ),
                num_threads=8,
            )
            commit_oid = commit.oid
            publication["repositories"][repo] = {
                "video_count": len(rows),
                "commit": commit_oid,
                "commit_url": commit.commit_url,
                "remote_verified": False,
                "source_review_commit": expected_review,
                "previous_main_commit": expected_base,
            }
            _atomic_json(main_publication_path, publication)
            print(f"promoted {repo}: {commit_oid}", flush=True)

        _verify_remote(api, repo=repo, revision=commit_oid, rows=rows)
        publication["repositories"][repo]["remote_verified"] = True
        _atomic_json(main_publication_path, publication)

    publication["completed_at"] = datetime.now(timezone.utc).isoformat()
    publication["all_remote_verified"] = all(
        row.get("remote_verified") for row in publication["repositories"].values()
    )
    _atomic_json(main_publication_path, publication)
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--correction-validation", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--original-publication", type=Path, required=True)
    parser.add_argument("--correction-publication", type=Path, required=True)
    parser.add_argument("--main-publication", type=Path, required=True)
    parser.add_argument("--review-branch", default="face-blur-2026-08-12")
    parser.add_argument("--expected-identity", default="oopsiedata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    promote(
        review_manifest=args.review_manifest.resolve(),
        correction_validation=args.correction_validation.resolve(),
        original_manifest=args.original_manifest.resolve(),
        release_path=args.release.resolve(),
        original_publication_path=args.original_publication.resolve(),
        correction_publication_path=args.correction_publication.resolve(),
        main_publication_path=args.main_publication.resolve(),
        review_branch=args.review_branch,
        expected_identity=args.expected_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
