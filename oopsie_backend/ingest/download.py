"""Download the Hugging Face files listed in an episode-index Parquet."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import polars as pl
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import build_hf_headers, validate_repo_id

from oopsie_backend.common.repos import data_root, is_local_import
from oopsie_backend.ingest.sync import matches_hub

REQUIRED_COLUMNS = ("dataset_repo", "repo_path")
OPTIONAL_COLUMNS = ("h5_sha256", "h5_git_oid", "videos")


def read_files(index: Path) -> tuple[dict[tuple[str, str], str | None], int]:
    """Read unique (repository, path) pairs and their optional content hashes."""
    schema = pl.read_parquet_schema(index)
    missing = set(REQUIRED_COLUMNS) - schema.keys()
    if missing:
        raise ValueError(f"index is missing columns: {', '.join(sorted(missing))}")
    if "videos" in schema and schema["videos"] != pl.Null:
        dtype = schema["videos"]
        if not isinstance(dtype, pl.List) or (
            dtype.inner != pl.Null
            and (not isinstance(dtype.inner, pl.Struct) or "path" not in dtype.inner.to_schema())
        ):
            raise ValueError("videos must be a list of structs with a path field")

    frame = pl.read_parquet(index, columns=[c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in schema])
    files = {}
    local_rows = 0
    for row in frame.iter_rows(named=True):
        repo = row["dataset_repo"]
        if not isinstance(repo, str) or repo.count("/") != 1:
            raise ValueError(f"expected a namespace/repo dataset ID, got {repo!r}")
        validate_repo_id(repo)
        if is_local_import(repo):
            local_rows += 1
            continue
        assets = [
            (row["repo_path"], row.get("h5_sha256") or row.get("h5_git_oid")),
            *((v["path"], v.get("sha256") or v.get("git_oid")) for v in row.get("videos") or [] if v is not None),
        ]
        for path, digest in assets:
            if (
                not isinstance(path, str)
                or "\\" in path
                or Path(path).drive
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                raise ValueError(f"expected a repository-relative file path, got {path!r} in {repo}")
            if digest and (not isinstance(digest, str) or len(digest) not in {40, 64}):
                raise ValueError(f"invalid content hash for {repo}/{path}")
            key = (repo, path)
            previous = files.get(key)
            if previous and digest and previous != digest and len(previous) == len(digest):
                raise ValueError(f"conflicting content hashes for {repo}/{path}")
            if previous is None or (digest and len(digest) == 64):
                files[key] = digest or None
    return files, local_rows


def matches_digest(path: Path, digest: str) -> bool:
    # Indexed sizes describe the mirror at indexing time. The Hub hash identifies the remote bytes.
    return matches_hub(path, {
        "size": path.stat().st_size,
        "sha256": digest if len(digest) == 64 else None,
        "git_oid": digest,
    })


def download_file(
    root: Path, repo: str, path: str, digest: str | None, revision: str | None, headers: dict[str, str],
) -> bool:
    """Verify a local copy or atomically install the download. Return whether it was downloaded."""
    dest = root / repo / path
    if not dest.resolve().is_relative_to(root):
        raise ValueError("destination resolves outside the data directory")
    url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
    if digest is None:
        metadata = get_hf_file_metadata(url, headers=headers)
        digest = metadata.etag
        if not digest or len(digest) not in {40, 64} or not metadata.commit_hash:
            raise ValueError("the Hub did not provide a supported file hash and commit")
        url = hf_hub_url(repo, path, repo_type="dataset", revision=metadata.commit_hash)
    if dest.is_file() and matches_digest(dest, digest):
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".oopsie-", suffix=".part", delete=False) as f:
        partial = Path(f.name)
    try:
        for attempt in range(3):
            try:
                with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=30) as response:
                    response.raise_for_status()
                    with partial.open("wb") as out:
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            out.write(chunk)
                break
            except httpx.HTTPError as exc:
                if attempt == 2 or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code not in {408, 429, 500, 502, 503, 504}
                ):
                    raise
                time.sleep(2**attempt)
        if not matches_digest(partial, digest):
            raise ValueError("download disagrees with the expected content hash")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="Parquet with dataset_repo and repo_path columns")
    parser.add_argument("--data-dir", required=True, type=Path, help="Download into <data-dir>/<namespace>/<repo>")
    parser.add_argument("--revisions", type=Path, help="Pin repo commits using a release snapshot.json")
    parser.add_argument("--workers", type=int, default=8, help="Parallel file downloads (default: 8)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        files, local_rows = read_files(args.index)
        revisions = {}
        if args.revisions:
            snapshot = json.loads(args.revisions.read_text())
            revisions = {r["repo_id"]: r["commit"] for r in snapshot["datasets"] if r.get("commit")}
            missing = {repo for repo, _ in files} - revisions.keys()
            if missing:
                raise ValueError(f"snapshot is missing a commit for: {', '.join(sorted(missing))}")
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
        parser.error(str(exc))

    root = data_root(args.data_dir).resolve()
    if local_rows:
        print(f"Skipped {local_rows} local import row(s)", file=sys.stderr)
    print(f"Checking {len(files)} file(s) in {root}", file=sys.stderr)
    headers = build_hf_headers()
    downloaded = verified = failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {
            pool.submit(download_file, root, repo, path, digest, revisions.get(repo), headers): (repo, path)
            for (repo, path), digest in files.items()
        }
        for done, task in enumerate(as_completed(tasks), 1):
            try:
                if task.result():
                    downloaded += 1
                else:
                    verified += 1
            except (OSError, ValueError, httpx.HTTPError) as exc:
                failed += 1
                repo, path = tasks[task]
                response = getattr(exc, "response", None)
                detail = f"HTTP {response.status_code}" if response is not None else type(exc).__name__
                if isinstance(exc, (OSError, ValueError)):
                    detail = str(exc)
                print(f"{repo}/{path}: {detail}", file=sys.stderr)
            if done % 100 == 0:
                print(f"Processed {done}/{len(files)} files", file=sys.stderr)
    print(f"{downloaded} downloaded, {verified} already verified, {failed} failed", file=sys.stderr)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
