"""Hub upload times of the indexed episodes, from the git history of their repos.

``hub_uploaded_at`` is the date of the commit that first added an episode's h5 to the
default branch of its Hub repo, and ``hub_upload_commit`` is that commit. Each repo's
history (commits and trees, no file contents) is kept as a bare clone in
``<mirror>/.hub_history/`` and fetched incrementally. Repos that fail to fetch keep
their earlier values.

Usage:
    uv run python -m oopsie_backend.ingest.upload_times --data-dir <mirror> --index episode_index.parquet
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import polars as pl

from oopsie_backend.common.h5 import H5_SUFFIXES
from oopsie_backend.common.repos import DEFAULT_DATA_DIR, data_root
from oopsie_backend.index.store import DEFAULT_INDEX, KEY, read_index, update_columns

HISTORY_DIR = ".hub_history"
COLUMNS = ["hub_uploaded_at", "hub_upload_commit"]
SCHEMA = {KEY: pl.String, "hub_uploaded_at": pl.Datetime("us", "UTC"), "hub_upload_commit": pl.String}


def _git(args: list[str], token: str | None = None) -> str:
    auth = ["-c", f"http.extraHeader=Authorization: Bearer {token}"] if token else []
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *auth, *args], env=env, check=True, capture_output=True, text=True
    ).stdout


def fetch_history(dataset_repo: str, root: Path, token: str | None) -> Path:
    """Bare clone of ``dataset_repo`` without file contents, with ``main`` at the Hub's default branch."""
    repo = root / f"{dataset_repo.split('/', 1)[1]}.git"
    if not repo.is_dir():
        url = f"https://huggingface.co/datasets/{dataset_repo}"
        _git(["clone", "--quiet", "--bare", "--filter=blob:none", url, str(repo)], token)
    _git(["-C", str(repo), "fetch", "--quiet", "--filter=blob:none", "origin", "+HEAD:refs/heads/main"], token)
    return repo


def first_uploads(repo: Path) -> dict[str, tuple[str, datetime]]:
    """Path -> ``(commit, date)`` of the first commit on ``main`` that added it, for every h5 ever added.

    Merged pull requests count from their merge commit.
    """
    log = _git([
        "-C", str(repo), "log", "main", "-m", "--first-parent", "--no-renames", "--diff-filter=A",
        "--name-only", "--format=%x00%H %cI", "--", *(f":(icase)*{s}" for s in H5_SUFFIXES),
    ])
    out = {}
    for block in log.split("\0")[1:]:
        header, *paths = block.strip().splitlines()
        commit, date = header.split()
        for path in filter(None, paths):
            out[path] = (commit, datetime.fromisoformat(date))  # newest first, so the oldest add wins
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Mirror root")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--only", nargs="+", default=None, help="Repo names to update (default: every repo on the Hub)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF read token (default: $HF_TOKEN)")
    parser.add_argument("--parallel-repos", type=int, default=4)
    args = parser.parse_args(argv)

    index = read_index(args.index, columns=["dataset_repo", "repo_path", "on_hub"])
    repos = sorted(index.filter(pl.col("on_hub"))["dataset_repo"].unique())
    if args.only:
        repos = [r for r in repos if r.split("/", 1)[1] in args.only]
    root = data_root(args.data_dir) / HISTORY_DIR
    root.mkdir(parents=True, exist_ok=True)

    def run(dataset_repo: str) -> tuple[str, dict | None]:
        try:
            return dataset_repo, first_uploads(fetch_history(dataset_repo, root, args.token))
        except subprocess.CalledProcessError as exc:
            print(f"{dataset_repo}: {exc.stderr.strip() or exc}", file=sys.stderr)
            return dataset_repo, None

    with ThreadPoolExecutor(args.parallel_repos) as pool:
        uploads = {r: u for r, u in pool.map(run, repos) if u is not None}
    rows = []
    for key, dataset_repo, repo_path in (
        index.filter(pl.col("dataset_repo").is_in(list(uploads))).select(KEY, "dataset_repo", "repo_path").iter_rows()
    ):
        commit, date = uploads[dataset_repo].get(repo_path, (None, None))
        rows.append({KEY: key, "hub_uploaded_at": date, "hub_upload_commit": commit})
    update_columns(args.index, pl.DataFrame(rows, schema=SCHEMA), COLUMNS)
    dated = sum(r["hub_upload_commit"] is not None for r in rows)
    print(f"upload times: {dated} of {len(rows)} episodes in {len(uploads)} repo(s) dated", file=sys.stderr)
    failed = len(repos) - len(uploads)
    if failed:
        print(f"upload times: {failed} repo(s) failed; they keep their earlier values", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
