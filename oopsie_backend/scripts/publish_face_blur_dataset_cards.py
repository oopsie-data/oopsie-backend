"""Publish dataset cards that expose the face-blurred repository tag."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


TAG = "face-blurred"
CARD_PATH = "README.md"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _card(repo: str, video_count: int) -> bytes:
    name = repo.rsplit("/", 1)[-1]
    return (
        "---\n"
        "tags:\n"
        f"- {TAG}\n"
        "---\n\n"
        f"# {name}\n\n"
        f"This repository contains {video_count} privacy-processed videos tagged "
        f"`{TAG}`. The tag indicates that the repository contains face-blurred "
        "videos; it does not mean that every video in the repository was modified.\n\n"
        "[`video_tags.csv`](video_tags.csv) is the authoritative per-video index. "
        "Filter its `tag` column for `face-blurred`; `video_path` gives the exact "
        "repository-relative MP4 path. The versioned processing manifest and "
        "validation records are under "
        "[`face_blur_release/2026-08-12/`](face_blur_release/2026-08-12/).\n\n"
        "Raw originals remain preserved by the `pre-face-blur-2026-08-12` tag and "
        "in the access-controlled raw archive.\n"
    ).encode("utf-8")


def publish(
    *,
    tag_publication_path: Path,
    card_publication_path: Path,
    expected_identity: str,
) -> None:
    tag_publication = _load_json(tag_publication_path)
    if not tag_publication.get("all_remote_verified"):
        raise ValueError("per-video tag publication has not been remotely verified")
    api = HfApi()
    identity = api.whoami().get("name")
    if identity != expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {expected_identity!r}"
        )
    publication = (
        _load_json(card_publication_path)
        if card_publication_path.is_file()
        else {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tag": TAG,
            "repositories": {},
        }
    )

    for repo, tagged in sorted(tag_publication["repositories"].items()):
        expected_parent = tagged["commit"]
        value = _card(repo, int(tagged["video_count"]))
        previous = publication["repositories"].get(repo)
        head = api.repo_info(repo, repo_type="dataset", revision="main").sha
        if previous:
            if head != previous["commit"]:
                raise ValueError(
                    f"dataset-card main branch moved for {repo}: "
                    f"{head} != {previous['commit']}"
                )
            commit_oid = previous["commit"]
            if previous.get("remote_verified"):
                print(f"already published and verified: {repo}", flush=True)
                continue
        else:
            if head != expected_parent:
                raise ValueError(
                    f"refusing to write dataset card to changed main for {repo}: "
                    f"{head} != {expected_parent}"
                )
            if api.file_exists(
                repo, CARD_PATH, repo_type="dataset", revision=expected_parent
            ):
                raise ValueError(
                    f"refusing to replace an unexpected existing dataset card: {repo}"
                )
            commit = api.create_commit(
                repo,
                repo_type="dataset",
                revision="main",
                parent_commit=expected_parent,
                operations=[
                    CommitOperationAdd(path_in_repo=CARD_PATH, path_or_fileobj=value)
                ],
                commit_message="Expose face-blurred dataset tag",
                commit_description=(
                    "Adds repository-level discovery metadata and links it to the "
                    "authoritative per-video tag index. Video bytes are unchanged."
                ),
            )
            commit_oid = commit.oid
            publication["repositories"][repo] = {
                "video_count": int(tagged["video_count"]),
                "commit": commit_oid,
                "commit_url": commit.commit_url,
                "previous_main_commit": expected_parent,
                "remote_verified": False,
            }
            _atomic_json(card_publication_path, publication)
            print(f"published card for {repo}: {commit_oid}", flush=True)

        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=commit_oid,
                filename=CARD_PATH,
                token=api.token,
                force_download=True,
            )
        )
        if downloaded.read_bytes() != value:
            raise ValueError(f"remote dataset card differs from generated card: {repo}")
        info = api.repo_info(repo, repo_type="dataset", revision=commit_oid)
        if TAG not in info.tags:
            raise ValueError(f"Hub did not expose {TAG!r} for {repo}: {info.tags}")
        publication["repositories"][repo]["remote_verified"] = True
        _atomic_json(card_publication_path, publication)
        print(f"verified Hub tag for {repo}", flush=True)

    publication["completed_at"] = datetime.now(timezone.utc).isoformat()
    publication["all_remote_verified"] = all(
        item.get("remote_verified") for item in publication["repositories"].values()
    )
    _atomic_json(card_publication_path, publication)
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-publication", type=Path, required=True)
    parser.add_argument("--card-publication", type=Path, required=True)
    parser.add_argument("--expected-identity", default="oopsiedata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publish(
        tag_publication_path=args.tag_publication.resolve(),
        card_publication_path=args.card_publication.resolve(),
        expected_identity=args.expected_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
