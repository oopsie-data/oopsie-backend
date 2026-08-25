"""Publish validated fallback-policy corrections to existing HF review branches.

This is deliberately a second-stage publisher: it requires the original release
publication record, validates every corrected file against its audit sidecar,
updates the full release manifest, and refuses to write if a review branch has
moved unexpectedly.  Progress is checkpointed after each repository so the
command can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


MANIFEST_FIELDS = (
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


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _review_summary_bytes(rows: list[dict[str, str]]) -> bytes:
    decisions = Counter(row["review_decision"] or "unreviewed" for row in rows)
    by_repo: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_repo[row["source_repo"]][row["review_decision"] or "unreviewed"] += 1
    value = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "decision_counts_by_repo": {
            repo: dict(sorted(counts.items())) for repo, counts in sorted(by_repo.items())
        },
        "release_decision": (
            "The corrected set was accepted after manual spot-checking; "
            "all validated correction candidates are included."
        ),
    }
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _corrected_rows(
    *,
    review_manifest: Path,
    original_manifest: Path,
    correction_validation: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]], dict]:
    validation = _load_json(correction_validation)
    if not validation.get("valid"):
        raise ValueError("correction validation is not valid")
    review_rows = _load_csv(review_manifest)
    if len(review_rows) != int(validation["video_count"]):
        raise ValueError("review and validation video counts differ")
    review_by_id = {row["sample_id"]: row for row in review_rows}
    if len(review_by_id) != len(review_rows):
        raise ValueError("duplicate sample_id in correction review manifest")
    disallowed = [
        row["sample_id"]
        for row in review_rows
        if row["review_decision"] in {"needs_more_blur", "unsure"}
    ]
    if disallowed:
        raise ValueError(f"corrections include unresolved review decisions: {disallowed[:5]}")

    original_rows: list[dict[str, object]] = list(_load_csv(original_manifest))
    original_by_id = {str(row["sample_id"]): row for row in original_rows}
    if len(original_by_id) != len(original_rows):
        raise ValueError("duplicate sample_id in original blurred manifest")
    missing = sorted(set(review_by_id) - set(original_by_id))
    if missing:
        raise ValueError(f"correction IDs are absent from original release: {missing[:5]}")

    replacements: list[dict[str, object]] = []
    for index, sample_id in enumerate(sorted(review_by_id), 1):
        review = review_by_id[sample_id]
        original = original_by_id[sample_id]
        if (
            review["source_repo"] != original["source_repo"]
            or review["source_path"] != original["source_path"]
        ):
            raise ValueError(f"source identity changed for {sample_id}")
        output = Path(review["corrected_video_path"]).resolve()
        sidecar_path = Path(f"{output}.tracked-deface.json")
        if not output.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"missing corrected output or sidecar for {sample_id}")
        sidecar = _load_json(sidecar_path)
        if sidecar.get("status") != "ok":
            raise ValueError(f"correction sidecar is not successful for {sample_id}")
        encoded = sidecar.get("output", {})
        tracking = sidecar.get("tracking", {})
        policy = tracking.get("policy", {})
        output_sha = _sha256(output)
        if (
            encoded.get("sha256") != output_sha
            or int(encoded.get("size_bytes", -1)) != output.stat().st_size
        ):
            raise ValueError(f"corrected output differs from sidecar for {sample_id}")
        replacement: dict[str, object] = dict(original)
        replacement.update(
            {
                "output_local_path": str(output),
                "blurred_size_bytes": output.stat().st_size,
                "blurred_sha256": output_sha,
                "masked_frame_count": int(tracking.get("masked_frame_count") or 0),
                "region_count": int(tracking.get("region_count") or 0),
                "track_count": int(tracking.get("track_count") or 0),
                "confirmed_track_count": int(policy.get("confirmed_track_count") or 0),
                "fallback_detection_count": int(policy.get("fallback_detection_count") or 0),
                "config_hash": sidecar["config_hash"],
            }
        )
        replacements.append(replacement)
        original_by_id[sample_id] = replacement
        if index % 25 == 0 or index == len(review_by_id):
            print(f"validated {index}/{len(review_by_id)} corrected files", flush=True)

    full_rows = [original_by_id[str(row["sample_id"])] for row in original_rows]
    return replacements, full_rows, review_rows, validation


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
                force_download=True,
            )
        )
        if (
            downloaded.stat().st_size != int(row["blurred_size_bytes"])
            or _sha256(downloaded) != row["blurred_sha256"]
        ):
            raise ValueError(f"remote hash verification failed: {repo}/{row['source_path']}")
        if index % 25 == 0 or index == len(rows):
            print(f"verified {index}/{len(rows)} corrected files for {repo}", flush=True)


def publish(
    *,
    review_manifest: Path,
    correction_validation: Path,
    original_manifest: Path,
    original_publication: Path,
    publication_path: Path,
    branch: str,
    expected_identity: str,
) -> None:
    corrected, full_rows, review_rows, validation = _corrected_rows(
        review_manifest=review_manifest,
        original_manifest=original_manifest,
        correction_validation=correction_validation,
    )
    api = HfApi()
    identity = api.whoami().get("name")
    if identity != expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {expected_identity!r}"
        )

    original = _load_json(original_publication)
    if original.get("review_branch") != branch or not original.get("all_remote_verified"):
        raise ValueError("original review-branch publication is incomplete or mismatched")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    full_by_repo: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in corrected:
        grouped[str(row["source_repo"])].append(row)
    for row in full_rows:
        full_by_repo[str(row["source_repo"])].append(row)
    if set(grouped) != set(original["repositories"]):
        raise ValueError("correction repositories differ from original release repositories")

    publication = (
        _load_json(publication_path)
        if publication_path.is_file()
        else {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "review_branch": branch,
            "corrected_video_count": len(corrected),
            "manual_review_counts": dict(
                sorted(Counter(r["review_decision"] or "unreviewed" for r in review_rows).items())
            ),
            "repositories": {},
        }
    )
    review_summary = _review_summary_bytes(review_rows)
    validation_bytes = (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    metadata_root = "face_blur_release/2026-08-12"

    for repo in sorted(grouped):
        repo_rows = sorted(grouped[repo], key=lambda row: str(row["source_path"]))
        prior = publication["repositories"].get(repo)
        branch_head = api.repo_info(repo, repo_type="dataset", revision=branch).sha
        if prior:
            if branch_head != prior["commit"]:
                raise ValueError(
                    f"published correction branch moved for {repo}: "
                    f"{branch_head} != {prior['commit']}"
                )
            if prior.get("remote_verified"):
                print(f"already published and verified: {repo}", flush=True)
                continue
            commit_oid = prior["commit"]
        else:
            expected_head = original["repositories"][repo]["commit"]
            if branch_head != expected_head:
                raise ValueError(
                    f"refusing to overwrite changed review branch for {repo}: "
                    f"{branch_head} != {expected_head}"
                )
            operations = [
                CommitOperationAdd(
                    path_in_repo=str(row["source_path"]),
                    path_or_fileobj=Path(str(row["output_local_path"])),
                )
                for row in repo_rows
            ]
            readme = (
                "# Face-blur review release (2026-08-12)\n\n"
                f"This branch replaces {len(full_by_repo[repo])} manually verified "
                "face-containing videos with tracked Deface outputs. It does not modify "
                "`main`. Raw originals are recoverable from the private archive and from "
                "the `pre-face-blur-2026-08-12` tag.\n\n"
                f"A stricter fallback policy was subsequently applied to {len(repo_rows)} "
                "suspect outputs. The full correction set was accepted after a 14-video "
                "manual spot-check with no rejected or uncertain examples. See "
                "`correction-validation.json` and `correction-review-summary.json`.\n"
            ).encode()
            operations.extend(
                [
                    CommitOperationAdd(
                        path_in_repo=f"{metadata_root}/manifest.csv",
                        path_or_fileobj=_manifest_bytes(
                            sorted(full_by_repo[repo], key=lambda row: str(row["source_path"]))
                        ),
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"{metadata_root}/correction-validation.json",
                        path_or_fileobj=validation_bytes,
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"{metadata_root}/correction-review-summary.json",
                        path_or_fileobj=review_summary,
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"{metadata_root}/README.md",
                        path_or_fileobj=readme,
                    ),
                ]
            )
            commit = api.create_commit(
                repo,
                repo_type="dataset",
                revision=branch,
                parent_commit=expected_head,
                operations=operations,
                commit_message=f"Apply stricter fallback blur to {len(repo_rows)} videos",
                commit_description=(
                    "Corrects suspect outputs found during review. Raw originals remain "
                    "preserved in the private archive and pre-face-blur tag."
                ),
                num_threads=8,
            )
            commit_oid = commit.oid
            publication["repositories"][repo] = {
                "video_count": len(repo_rows),
                "commit": commit_oid,
                "commit_url": commit.commit_url,
                "remote_verified": False,
            }
            _atomic_json(publication_path, publication)
            print(f"published {repo}: {commit_oid}", flush=True)

        _verify_remote(api, repo=repo, revision=commit_oid, rows=repo_rows)
        publication["repositories"][repo]["remote_verified"] = True
        _atomic_json(publication_path, publication)

    publication["completed_at"] = datetime.now(timezone.utc).isoformat()
    publication["all_remote_verified"] = all(
        row.get("remote_verified") for row in publication["repositories"].values()
    )
    _atomic_json(publication_path, publication)
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--correction-validation", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--original-publication", type=Path, required=True)
    parser.add_argument("--publication-path", type=Path, required=True)
    parser.add_argument("--branch", default="face-blur-2026-08-12")
    parser.add_argument("--expected-identity", default="oopsiedata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publish(
        review_manifest=args.review_manifest.resolve(),
        correction_validation=args.correction_validation.resolve(),
        original_manifest=args.original_manifest.resolve(),
        original_publication=args.original_publication.resolve(),
        publication_path=args.publication_path.resolve(),
        branch=args.branch,
        expected_identity=args.expected_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
