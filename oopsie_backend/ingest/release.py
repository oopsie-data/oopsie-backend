"""Data releases: the mirror's Hub commits, the episode index and the annotation decisions, on the Hub.

A release lives in the private dataset repo ``RELEASE_REPO`` as a folder named after it
(by default the sync date, e.g. ``2026-11-08``), tagged with the same name:

    <name>/snapshot.json           a clean sync report: the Hub commit of every synced repo
    <name>/episode_index.parquet   the index built from exactly that data
    <name>/annotation/             eval proposal, train sample, assignments, annotator list,
                                   proposals and annotation stores, with their .inputs records
    <name>/configs/                the backend's configs/ when the release was made
    <name>/manifest.json           backend commit, annotation repo commits, sha256 of every file
    <name>/README.md               how to reproduce it

``publish`` refuses unless the snapshot is clean, every index episode is in it, the index
matches the sync record, and the sync record matches the Hub at each pinned commit.
``oopsie ingest sync --release <name> --data-dir <dir>`` downloads a release and syncs its data.

Usage:
    uv run oopsie ingest release publish --snapshot /corral/utexas/IRI26024/oopsiedata/snapshots/2026-11-08.json
    uv run oopsie ingest release list
    uv run oopsie pipeline run release --set release_snapshot=<snapshot.json>
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yaml
from huggingface_hub import HfApi, snapshot_download

from oopsie_backend.common.hf import hub_files
from oopsie_backend.common.io import now_timestamps
from oopsie_backend.common.repos import CONFIG_DIR, DEFAULT_DATA_DIR, IMPORTS_PREFIX, annotation_dir, data_root
from oopsie_backend.splits.eval_split import ID_SPLIT, OD_TASK_SPLIT

RELEASE_REPO = "OopsieData-Submissions/oopsiedata-releases"
DEFAULT_INDEX = Path("/corral/utexas/IRI26024/episode_index/episode_index.parquet")
BACKEND_DIR = CONFIG_DIR.parent
# Annotation decisions copied into a release; the large working folders (annotation_repo,
# review_bundles) are rebuilt from these and the annotation repos pinned in the manifest.
ANNOTATION_FILES = [
    "annotators.txt",
    "eval_proposal.json",
    "train_annotation_sample.json",
    "eval_assignment",
    "train_assignment",
    "proposals",
]
ANNOTATION_STORE_GLOBS = ["*_annotations.jsonl", "*_annotations_summary.json"]
# The release files a partial download (``sync --split``) fetches; it skips annotation/ and configs/.
INDEX_FILES = ["snapshot.json", "manifest.json", "episode_index.parquet"]
# Subsets of a release that ``oopsie ingest sync --release NAME --split SPLIT`` downloads.
# ``excluded`` and ``od_task_od_scene`` episodes are in neither.
SPLITS = {
    "test": pl.col("eval_split").is_in([ID_SPLIT, OD_TASK_SPLIT]),
    "train": pl.col("eval_split").is_null(),
}


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_name(snapshot: dict) -> str:
    """The sync date of the snapshot, in Austin time."""
    synced = snapshot.get("synced_at") or {}
    stamp = synced.get("cdt") or synced.get("utc")
    if not stamp:
        raise ValueError("the snapshot has no synced_at; pass --name")
    return stamp[:10]


def check_snapshot(snapshot: dict) -> list[str]:
    problems = []
    if snapshot.get("dry_run"):
        problems.append("the snapshot is from a dry run")
    if snapshot.get("datasets_failed"):
        problems.append(f"{snapshot['datasets_failed']} repo(s) failed to sync")
    if snapshot.get("files_to_download"):
        problems.append(f"{snapshot['files_to_download']} file(s) were left to download")
    if snapshot.get("split"):
        problems.append(f"the snapshot is a partial download (--split {snapshot['split']})")
    missing = [d["repo_id"] for d in snapshot["datasets"] if not d.get("commit")]
    if missing:
        problems.append(f"repo(s) without a commit: {', '.join(missing)}")
    return problems


def split_files(index_path: Path, split: str) -> tuple[dict[str, dict[str, str]], int]:
    """The files of the episodes in ``split``: ``{repo id: {path: git oid}}`` and the episode count.

    Each episode contributes its h5 and the mp4 of every camera; the git oids are the ones the index recorded.
    """
    rows = pl.read_parquet(index_path, columns=["dataset_repo", "repo_path", "h5_git_oid", "videos", "eval_split"]).filter(
        SPLITS[split] & ~pl.col("dataset_repo").str.starts_with(f"{IMPORTS_PREFIX}/")
    )
    files: dict[str, dict[str, str]] = {}
    for r in rows.iter_rows(named=True):
        repo = files.setdefault(r["dataset_repo"], {})
        repo[r["repo_path"]] = r["h5_git_oid"]
        for v in r["videos"] or []:
            repo[v["path"]] = v["git_oid"]
    return files, rows.height


def check_index(index: pl.DataFrame, state: pl.DataFrame, repos: set[str]) -> list[str]:
    """Every Hub episode of the index is in the snapshot, on disk, and recorded with the index's sha256."""
    problems = []
    hub = index.filter(~pl.col("dataset_repo").str.starts_with(f"{IMPORTS_PREFIX}/"))
    outside = sorted(set(hub["dataset_repo"].unique()) - repos)
    if outside:
        problems.append(f"index repo(s) missing from the snapshot: {', '.join(outside)}")
    not_on_disk = hub.filter(~pl.col("on_disk").fill_null(False))
    if not_on_disk.height:
        problems.append(f"{not_on_disk.height} index episode(s) are not on disk")
    joined = hub.join(state, left_on=["dataset_repo", "repo_path"], right_on=["dataset_repo", "path"], how="left")
    unrecorded = joined.filter(pl.col("sha256").is_null())
    if unrecorded.height:
        problems.append(f"{unrecorded.height} index episode(s) are not in the sync record")
    differ = joined.filter(pl.col("sha256").is_not_null() & (pl.col("sha256") != pl.col("h5_sha256")))
    if differ.height:
        by_repo = differ.group_by("dataset_repo").len().sort("dataset_repo").iter_rows()
        problems.append(
            "index episode(s) whose file changed since indexing: " + ", ".join(f"{r} ({n})" for r, n in by_repo)
        )
    return problems


def check_hub(api: HfApi, commits: dict[str, str], state: pl.DataFrame, workers: int = 8) -> list[str]:
    """The sync record of each repo lists exactly the files of its pinned commit."""

    def check(item: tuple[str, str]) -> str | None:
        repo_id, commit = item
        try:
            listing = hub_files(api, repo_id, revision=commit)
        except Exception as exc:
            return f"{repo_id}: cannot list commit {commit[:12]}: {type(exc).__name__}: {exc}"
        recorded = dict(state.filter(pl.col("dataset_repo") == repo_id).select("path", "git_oid").iter_rows())
        hub = {path: f["git_oid"] for path, f in listing.items()}
        if recorded == hub:
            return None
        changed = sum(recorded[p] != hub[p] for p in recorded.keys() & hub.keys())
        return (
            f"{repo_id}: the sync record differs from commit {commit[:12]}: {len(hub.keys() - recorded.keys())} "
            f"file(s) not synced, {len(recorded.keys() - hub.keys())} not in the commit, {changed} changed"
        )

    with ThreadPoolExecutor(workers) as pool:
        return [p for p in pool.map(check, sorted(commits.items())) if p]


def backend_commit() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(BACKEND_DIR), *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def annotation_repo_commits(api: HfApi) -> dict[str, str]:
    cfg = yaml.safe_load((CONFIG_DIR / "annotation_pipeline.yaml").read_text())
    repos = [cfg[k] for k in ("repo_id", "returns_repo_id") if cfg.get(k)]
    return {r: api.dataset_info(r).sha for r in repos}


def readme(name: str, manifest: dict) -> str:
    backend = manifest["backend"]
    return f"""# Oopsie data release {name}

The Oopsie mirror as synced on {name}: {manifest["repos"]} repos of the OopsieData-Submissions org,
each pinned to a Hub commit, the episode index built from exactly that data
({manifest["index_rows"]} episodes), and the annotation decisions made on it.

| File | Contents |
|---|---|
| `snapshot.json` | the sync report: the Hub commit of every repo (`datasets[].commit`) |
| `episode_index.parquet` | the episode index; rows are keyed by `dataset_repo` + `repo_path` |
| `annotation/` | eval proposal, train sample, annotator assignments, proposals and annotation stores |
| `configs/` | the oopsie-backend configs when the release was made |
| `manifest.json` | backend commit, annotation repo commits and the sha256 of every file |

## Reproduce

With [oopsie-backend](https://github.com/oopsie-data/oopsie-backend) at commit `{backend["commit"]}` or later, and an HF token
that can read the org:

```bash
export HF_TOKEN=hf_...
uv run oopsie ingest sync --release {name} --data-dir /path/to/data
```

This downloads the release to `<data dir>/releases/{name}/`, syncs every repo at its commit into
`<data dir>/OopsieData-Submissions/`, and copies `annotation/` to `<data dir>/annotation/`. The
index is `<data dir>/releases/{name}/episode_index.parquet`. Without oopsie-backend, download each
repo with `hf download <repo> --repo-type dataset --revision <commit>`.

The annotation working copies and returns are in the Hub repos listed under
`annotation_repos` in `manifest.json`, at the commits recorded there.
"""


def publish(
    snapshot_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    index_path: Path = DEFAULT_INDEX,
    name: str | None = None,
    repo_id: str = RELEASE_REPO,
    token: str | None = None,
    check_hub_commits: bool = True,
    dry_run: bool = False,
) -> str:
    """Check and upload a release; returns its Hub commit (or ``"dry-run"``)."""
    root = data_root(data_dir)
    snapshot = json.loads(snapshot_path.read_text())
    name = name or release_name(snapshot)
    api = HfApi(token=token)

    problems = check_snapshot(snapshot)
    commits = {d["repo_id"]: d["commit"] for d in snapshot["datasets"] if d.get("commit")}
    _log(f"release {name}: checking the index against the sync record")
    index = pl.read_parquet(index_path)
    state = pl.read_parquet(root / ".sync_state.parquet", columns=["dataset_repo", "path", "git_oid", "sha256"])
    problems += check_index(index, state, set(commits))
    if check_hub_commits:
        _log(f"release {name}: checking the sync record against {len(commits)} pinned Hub commit(s)")
        problems += check_hub(api, commits, state)
    if problems:
        raise RuntimeError(f"release {name} not published:\n  " + "\n  ".join(problems))

    if not dry_run:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        if any(t.name == name for t in api.list_repo_refs(repo_id, repo_type="dataset").tags):
            raise RuntimeError(f"release {name} already exists in {repo_id}; releases are never replaced")

    with tempfile.TemporaryDirectory(prefix=f"oopsie-release-{name}-") as tmp:
        out = Path(tmp) / name
        out.mkdir()
        shutil.copyfile(snapshot_path, out / "snapshot.json")
        shutil.copyfile(index_path, out / "episode_index.parquet")
        ann, ann_out = annotation_dir(root), out / "annotation"
        ann_out.mkdir()
        for item in ANNOTATION_FILES:
            for src in [ann / item, ann / f"{item}.inputs"]:
                if src.is_dir():
                    shutil.copytree(src, ann_out / src.name)
                elif src.is_file():
                    shutil.copyfile(src, ann_out / src.name)
        for pattern in ANNOTATION_STORE_GLOBS:
            for src in ann.glob(pattern):
                shutil.copyfile(src, ann_out / src.name)
        shutil.copytree(CONFIG_DIR, out / "configs")

        files = {str(p.relative_to(out)): _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
        manifest = {
            "name": name,
            "created_at": now_timestamps(),
            "created_by": os.environ.get("USER") or getpass.getuser(),
            "repo": repo_id,
            "repos": len(commits),
            "index_rows": index.height,
            "backend": backend_commit(),
            "annotation_repos": annotation_repo_commits(api),
            "files": files,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        (out / "README.md").write_text(readme(name, manifest))
        if manifest["backend"]["dirty"]:
            _log("warning: oopsie-backend has uncommitted changes; the manifest records HEAD")
        if dry_run:
            _log(f"dry run: release {name} would upload {len(files) + 2} file(s) to {repo_id}/{name}")
            return "dry-run"
        _log(f"release {name}: uploading {len(files) + 2} file(s) to {repo_id}/{name}")
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=out, path_in_repo=name,
            commit_message=f"Release {name}",
        )
    api.create_tag(repo_id, repo_type="dataset", tag=name, revision=commit.oid, tag_message=f"Release {name}")
    _log(f"release {name}: published at {repo_id}, tag {name} ({commit.oid[:12]})")
    return commit.oid


def fetch(
    name: str, data_dir: Path, repo_id: str = RELEASE_REPO, token: str | None = None, index_only: bool = False
) -> Path:
    """Download release ``name`` into ``<data dir>/releases/<name>/``; returns that folder.

    Copies its ``annotation/`` files to ``<data dir>/annotation/`` where missing; existing ones are kept.
    With ``index_only``, downloads only the snapshot, manifest and index (``INDEX_FILES``).
    """
    root = data_root(data_dir)
    local = root / "releases"
    patterns = [f"{name}/{f}" for f in INDEX_FILES] if index_only else [f"{name}/*"]
    snapshot_download(
        repo_id, repo_type="dataset", revision=name, allow_patterns=patterns, local_dir=local, token=token
    )
    shutil.rmtree(local / ".cache", ignore_errors=True)
    out = local / name
    if not (out / "snapshot.json").is_file():
        raise FileNotFoundError(f"release {name} has no snapshot.json in {repo_id}")
    manifest = json.loads((out / "manifest.json").read_text())
    expected = {f: sha for f, sha in manifest["files"].items() if not index_only or f in INDEX_FILES}
    bad = [f for f, sha in expected.items() if _sha256(out / f) != sha]
    if bad:
        raise RuntimeError(f"release {name}: downloaded file(s) do not match the manifest: {', '.join(bad)}")
    if index_only:
        _log(f"release {name}: index downloaded to {out / 'episode_index.parquet'}")
        return out
    ann = annotation_dir(root)
    kept = []
    for src in sorted((out / "annotation").iterdir()):
        dest = ann / src.name
        if dest.exists():
            kept.append(src.name)
            continue
        ann.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dest)
    if kept:
        _log(f"release {name}: kept existing annotation file(s) in {ann}: {', '.join(kept)}")
    _log(f"release {name}: downloaded to {out}; index at {out / 'episode_index.parquet'}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=RELEASE_REPO, help="Release repo")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (default: $HF_TOKEN)")
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="Check and upload a release")
    pub.add_argument("--snapshot", type=Path, required=True, help="A clean sync report, e.g. a copy of .sync_report.json")
    pub.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Mirror root the snapshot describes")
    pub.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Episode index built from that mirror")
    pub.add_argument("--name", default=None, help="Release name and tag (default: the sync date)")
    pub.add_argument("--skip-hub-check", action="store_true", help="Do not compare the sync record with the Hub commits")
    pub.add_argument("--dry-run", action="store_true", help="Run the checks and build the release; upload nothing")
    sub.add_parser("list", help="List the published releases")
    args = parser.parse_args(argv)

    if args.command == "list":
        api = HfApi(token=args.token)
        for tag in sorted(api.list_repo_refs(args.repo, repo_type="dataset").tags, key=lambda t: t.name):
            commit = api.dataset_info(args.repo, revision=tag.name).sha
            print(f"{tag.name}  {commit[:12]}")
        return 0
    try:
        publish(
            args.snapshot, args.data_dir, args.index, args.name, args.repo, args.token,
            check_hub_commits=not args.skip_hub_check, dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        _log(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
