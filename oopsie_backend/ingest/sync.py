"""Mirror every dataset repo of the Hugging Face org into ``<data_dir>/<org>/<repo>/``.

Archive and infra repos of ``configs/repositories.yaml`` are skipped unless named
with ``--only``.

A file is downloaded when it is new, when its Hub blob changed, or when the local
copy no longer has the size and modification time recorded after its download.
That record is one table for the whole mirror, ``<data_dir>/.sync_state.parquet``;
nothing else is written next to the data. A file that is on disk but not in the
record is checked once: against the ``.cache/huggingface`` metadata that
``huggingface_hub`` downloads leave behind if it is current, else by hashing it.

``.cache/huggingface`` holds two files per data file. It is deleted from each repo
once every file of the repo is in the record (``--keep-hf-cache`` keeps it).

Files the Hub no longer has are listed in the report; ``--delete-removed`` deletes
them. ``--scan-local`` also walks the repo folders for files that are in neither
the Hub listing nor the record, and for repo folders the Hub no longer has.

Usage:
    export HF_TOKEN=hf_...
    uv run oopsie ingest sync --data-dir /corral/utexas/IRI26024/oopsiedata
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import build_hf_headers, disable_progress_bars

from oopsie_backend.common import interrupt
from oopsie_backend.common.hf import dataset_repo_size, hub_files, list_dataset_repos_for_org, parse_org_arg
from oopsie_backend.common.io import now_timestamps
from oopsie_backend.common.repos import (
    ARCHIVE_DATASETS,
    DEFAULT_DATA_DIR,
    EXTERNAL_DATASETS,
    HF_ORG,
    IGNORED_DIR_NAMES,
    INFRA_DATASETS,
    data_root,
)

STATE_NAME = ".sync_state.parquet"
REPORT_NAME = ".sync_report.json"
STATE_SCHEMA = {
    "dataset_repo": pl.String,
    "path": pl.String,
    "git_oid": pl.String,
    "sha256": pl.String,
    "size": pl.Int64,
    "mtime_ns": pl.Int64,
}
HF_CACHE = Path(".cache") / "huggingface"


def _log(message: str, level: int = 0) -> None:
    print("  " * level + message, file=sys.stderr)


def read_state(path: Path) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        return {}
    return {(r["dataset_repo"], r["path"]): r for r in pl.read_parquet(path).iter_rows(named=True)}


def write_state(state: dict[tuple[str, str], dict], path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pl.DataFrame(list(state.values()), schema=STATE_SCHEMA).sort("dataset_repo", "path").write_parquet(tmp)
    os.replace(tmp, path)


def _hash(path: Path, algorithm: str, prefix: bytes = b"") -> str:
    h = hashlib.new(algorithm, prefix)
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def matches_hub(path: Path, hub: dict) -> bool:
    """Whether the local file has the Hub content: sha256 for LFS files, the git blob id otherwise."""
    if path.stat().st_size != hub["size"]:
        return False
    if hub["sha256"]:
        return _hash(path, "sha256") == hub["sha256"]
    return _hash(path, "sha1", f"blob {hub['size']}\0".encode()) == hub["git_oid"]


def hf_metadata_matches(repo_dir: Path, rel: str, hub: dict) -> bool:
    """Whether ``huggingface_hub``'s download metadata vouches for the local file.

    The metadata holds the commit, the etag (sha256 for LFS files, the git blob id
    otherwise) and when it was written; it is current if the file was not modified since.
    """
    meta = repo_dir / HF_CACHE / "download" / f"{rel}.metadata"
    try:
        _, etag, timestamp = meta.read_text().splitlines()[:3]
        current = (repo_dir / rel).stat().st_mtime - 1 <= float(timestamp)
    except (OSError, ValueError):
        return False
    return current and etag.strip() in {hub["sha256"], hub["git_oid"]}


class Progress:
    """Logs ``<what> n/total`` at most every ``interval`` seconds."""

    def __init__(self, repo_id: str, interval: float = 30.0):
        self.repo_id, self.interval, self.last = repo_id, interval, time.monotonic()

    def tick(self, what: str, n: int, total: int) -> None:
        if time.monotonic() - self.last >= self.interval:
            self.last = time.monotonic()
            _log(f"{self.repo_id}: {what} {n}/{total}", level=1)


@dataclass
class RepoResult:
    repo_id: str
    seconds: float = 0.0
    downloaded: int = 0
    adopted: int = 0
    unchanged: int = 0
    pending: int = 0
    removed: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    error: str | None = None


class Mirror:
    def __init__(self, root: Path, token: str | None, workers: int, gid: int | None):
        self.root = root
        self.token = token
        self.api = HfApi(token=token)
        self.workers = workers
        self.gid = gid
        self.lock = threading.Lock()  # guards the shared sync record

    def _make_dirs(self, path: Path) -> None:
        missing = []
        while not path.exists():
            missing.append(path)
            path = path.parent
        for d in reversed(missing):
            d.mkdir(exist_ok=True)
            if self.gid is not None and d.stat().st_gid != self.gid:
                os.chown(d, -1, self.gid)
                d.chmod(d.stat().st_mode | 0o2070)

    def download(self, repo_id: str, revision: str, rel: str, hub: dict, dest: Path) -> None:
        """Download one file to a temporary name next to ``dest`` and rename it into place."""
        self._make_dirs(dest.parent)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        url = hf_hub_url(repo_id, rel, repo_type="dataset", revision=revision)
        try:
            with tmp.open("wb") as f:
                http_get(url, f, headers=build_hf_headers(token=self.token), expected_size=hub["size"])
            if self.gid is not None and tmp.stat().st_gid != self.gid:
                os.chown(tmp, -1, self.gid)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    def sync_repo(self, repo_id: str, state: dict, args: argparse.Namespace, download: bool) -> RepoResult:
        """Check every Hub file of a repo against ``state``; with ``download``, fetch the new and changed ones.

        Without ``download`` (and in a dry run), ``result.pending`` counts the files that need one.
        """
        result = RepoResult(repo_id)
        repo_dir = self.root / repo_id
        revision = self.api.dataset_info(repo_id).sha
        _log(f"{repo_id}: listing the Hub", level=1)
        listing = hub_files(self.api, repo_id, revision=revision)
        _log(f"{repo_id}: {len(listing)} file(s) on the Hub; checking local copies", level=1)
        progress = Progress(repo_id)

        def record(rel: str, hub: dict) -> None:
            st = (repo_dir / rel).stat()
            with self.lock:
                state[(repo_id, rel)] = {
                    "dataset_repo": repo_id, "path": rel, "git_oid": hub["git_oid"],
                    "sha256": hub["sha256"], "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                }

        def check(rel: str) -> str:
            """``unchanged``, ``adopted`` or ``download``."""
            hub, path, old = listing[rel], repo_dir / rel, state.get((repo_id, rel))
            try:
                st = path.stat()
            except FileNotFoundError:
                return "download"
            if old and old["git_oid"] == hub["git_oid"] and (old["size"], old["mtime_ns"]) == (st.st_size, st.st_mtime_ns):
                return "unchanged"
            if old and old["git_oid"] != hub["git_oid"]:
                return "download"
            if hf_metadata_matches(repo_dir, rel, hub) or matches_hub(path, hub):
                return "adopted"
            return "download"

        def fetch(rel: str) -> None:
            self.download(repo_id, revision, rel, listing[rel], repo_dir / rel)

        with ThreadPoolExecutor(self.workers) as pool:
            try:
                decisions = {}
                for rel, decision in zip(listing, pool.map(check, listing)):
                    decisions[rel] = decision
                    progress.tick("checked", len(decisions), len(listing))
                for rel, decision in decisions.items():
                    if decision == "unchanged":
                        result.unchanged += 1
                    elif decision == "adopted":
                        record(rel, listing[rel])
                        result.adopted += 1
                todo = [rel for rel, d in decisions.items() if d == "download"]
                if not download:
                    result.pending = len(todo)
                else:
                    futures = {pool.submit(fetch, rel): rel for rel in todo}
                    size = sum(listing[rel]["size"] for rel in todo)
                    if todo:
                        _log(f"{repo_id}: downloading {len(todo)} file(s), {size / 2**30:.1f} GiB", level=1)
                    for done, future in enumerate(as_completed(futures), 1):
                        rel = futures[future]
                        progress.tick("downloaded", done, len(todo))
                        try:
                            future.result()
                        except Exception as exc:
                            result.failed.append(f"{rel}: {type(exc).__name__}: {exc}")
                            continue
                        record(rel, listing[rel])
                        result.downloaded += 1
            except BaseException:
                # Ctrl+C: drop the queued checks and downloads; running ones finish.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        if download:
            return result
        with self.lock:
            gone = [k for k in state if k[0] == repo_id and k[1] not in listing]
        for key in gone:
            result.removed.append(key[1])
            if args.delete_removed and not args.dry_run:
                (repo_dir / key[1]).unlink(missing_ok=True)
                with self.lock:
                    del state[key]
        if args.scan_local:
            result.untracked = sorted(set(local_files(repo_dir)) - set(listing) - set(result.removed))
            if args.delete_removed and not args.dry_run:
                for rel in result.untracked:
                    (repo_dir / rel).unlink(missing_ok=True)
        return result


def prune_hf_locks(repo_dir: Path) -> int:
    """Delete the lock files and unfinished downloads of ``huggingface_hub``; returns how many."""
    n = 0
    for path in (repo_dir / HF_CACHE / "download").rglob("*"):
        if path.suffix in {".lock", ".incomplete"} and path.is_file():
            path.unlink(missing_ok=True)
            n += 1
    return n


def prune_hf_cache(repo_dir: Path) -> None:
    """Delete the ``.cache/huggingface`` folder of a repo, and ``.cache`` if it is then empty.

    The files are deleted from parallel threads, since each delete is a round trip to a network file system.
    """
    download = repo_dir / HF_CACHE / "download"
    if download.is_dir():
        with ThreadPoolExecutor(16) as pool:
            for dirpath, _, filenames in os.walk(download):
                pool.map(os.unlink, [os.path.join(dirpath, f) for f in filenames])
    shutil.rmtree(repo_dir / HF_CACHE, ignore_errors=True)
    try:
        (repo_dir / ".cache").rmdir()
    except OSError:
        pass


def local_files(repo_dir: Path) -> list[str]:
    """Repo-relative paths of the files in a repo folder, without hidden and ignored folders and temporary downloads."""
    out = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in IGNORED_DIR_NAMES]
        rel = Path(dirpath).relative_to(repo_dir)
        out += [(rel / f).as_posix() for f in filenames if not f.endswith(".part")]
    return out


class Saver:
    """Writes the sync record at most every ``interval`` seconds, then deletes the ``.cache`` of the repos it covers.

    A repo's ``.cache`` is deleted only after a write that includes the repo's entries.
    """

    def __init__(self, state: dict, path: Path, state_lock: threading.Lock, dry_run: bool, interval: float = 60.0):
        self.state, self.path, self.state_lock, self.dry_run, self.interval = state, path, state_lock, dry_run, interval
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self.to_prune: list[Path] = []

    def done(self, prune: Path | None) -> None:
        """Called when a repo is finished; ``prune`` is its folder if its ``.cache`` may go."""
        with self.lock:
            if prune is not None:
                self.to_prune.append(prune)
            due = time.monotonic() - self.last >= self.interval
        if due:
            self.save()

    def save(self) -> None:
        if self.dry_run:
            return
        with self.lock:
            with self.state_lock:
                snapshot = dict(self.state)
            write_state(snapshot, self.path)
            self.last = time.monotonic()
            to_prune, self.to_prune = self.to_prune, []
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(prune_hf_cache, to_prune))


def each_repo(run: Callable[..., None], repo_ids: list[str], parallel: int, download: bool) -> None:
    """Call ``run`` for every repo, ``parallel`` repos at a time."""
    with ThreadPoolExecutor(parallel) as pool:
        futures = [pool.submit(run, repo_id, download, len(repo_ids)) for repo_id in repo_ids]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def smallest_repos(repo_ids: list[str], n: int, token: str | None) -> list[str]:
    """The ``n`` smallest contributor repos that hold ``.h5`` files; for local test mirrors."""
    sized = []
    for repo_id in repo_ids:
        if repo_id in ARCHIVE_DATASETS | EXTERNAL_DATASETS:
            continue
        try:
            total, n_h5 = dataset_repo_size(repo_id, token)
        except RepositoryNotFoundError:
            continue
        if n_h5:
            sized.append((total, repo_id))
    return [repo_id for _, repo_id in sorted(sized)[:n]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", "--local-dir", type=Path, default=DEFAULT_DATA_DIR, help="Mirror root")
    parser.add_argument("--org", default=HF_ORG)
    parser.add_argument("--only", nargs="+", default=None, help="Repo names to sync, archive and infra repos included (default: every other repo of the org)")
    parser.add_argument("--smallest", type=int, default=None, help="Sync only the N smallest contributor repos with h5 files")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF read token (default: $HF_TOKEN)")
    parser.add_argument("--parallel-repos", type=int, default=4, help="Repos processed at the same time, one per worker")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads and checks within a repo")
    parser.add_argument("--group", default=None, help="Group of every created file and folder, e.g. G-829069")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be downloaded; writes nothing")
    parser.add_argument("--keep-hf-cache", action="store_true", help="Keep the .cache/huggingface folders")
    parser.add_argument("--scan-local", action="store_true", help="Also report local files and repo folders the Hub does not have")
    parser.add_argument("--no-download", action="store_true", help="Check, adopt and delete .cache folders only; download nothing")
    parser.add_argument("--state", type=Path, default=None, help=f"Sync record; the report is written next to it (default: <data dir>/{STATE_NAME})")
    parser.add_argument("--delete-removed", action="store_true", help="Delete local files the Hub no longer has")
    args = parser.parse_args(argv)

    os.umask(0o002)
    disable_progress_bars()
    # Worker threads in file system calls do not see KeyboardInterrupt, so a signal exits at once.
    # That is safe: the record is written whole, and unsaved repos keep their .cache for the next run.
    interrupt.install()
    gid = None
    if args.group:
        try:
            gid = grp.getgrnam(args.group).gr_gid
        except KeyError:
            _log(f"error: unknown group {args.group!r}")
            return 1

    if not args.token:
        _log("warning: no HF_TOKEN; the Hub lists only public repos")
    org = parse_org_arg(args.org)
    root = data_root(args.data_dir)
    listed = list_dataset_repos_for_org(org, args.token)
    if args.only:
        repo_ids = [r for r in listed if r.split("/", 1)[1] in set(args.only)]
    else:
        repo_ids = [r for r in listed if r not in ARCHIVE_DATASETS | INFRA_DATASETS]
    if args.smallest is not None:
        repo_ids = smallest_repos(repo_ids, args.smallest, args.token)
    if not repo_ids:
        _log(f"error: no datasets found for {org!r}; check the org and HF_TOKEN read access")
        return 1

    if not args.keep_hf_cache and not args.dry_run:
        # Frees about one file per data file before the sync writes anything.
        with ThreadPoolExecutor(16) as pool:
            pruned = sum(pool.map(prune_hf_locks, [root / repo_id for repo_id in repo_ids]))
        _log(f"deleted {pruned} huggingface_hub lock file(s) and unfinished download(s)")

    state_path = args.state or root / STATE_NAME
    state = read_state(state_path)
    mirror = Mirror(root, args.token, args.workers, gid)
    results: dict[str, RepoResult] = {}
    saver = Saver(state, state_path, mirror.lock, dry_run=args.dry_run)
    log_lock = threading.Lock()
    done = {"n": 0}

    def run(repo_id: str, download: bool, total: int) -> None:
        started = time.perf_counter()
        try:
            result = mirror.sync_repo(repo_id, state, args, download)
        except RepositoryNotFoundError:
            result = RepoResult(repo_id, error="not found")
        except Exception as exc:
            result = RepoResult(repo_id, error=f"{type(exc).__name__}: {exc}")
        result.seconds = time.perf_counter() - started
        prune = not (args.keep_hf_cache or result.error or result.failed)
        saver.done(root / repo_id if prune else None)
        if download:
            first = results[repo_id]
            first.downloaded, first.pending = result.downloaded, 0
            first.failed, first.error = result.failed, result.error
            first.seconds += result.seconds
            summary = result.error or f"downloaded {result.downloaded}, failed {len(result.failed)}"
        else:
            results[repo_id] = result
            summary = result.error or (
                f"adopted {result.adopted}, unchanged {result.unchanged}, to download {result.pending}, "
                f"removed on Hub {len(result.removed)}"
            )
        with log_lock:
            done["n"] += 1
            _log(f"[{done['n']}/{total}] {repo_id}: {summary} ({result.seconds:.0f}s)")
        for line in result.failed[:5]:
            _log(line, level=1)

    # Checking every repo first deletes the .cache folders, about one file per data file, before
    # downloads add files.
    _log(f"{'Dry run of' if args.dry_run else 'Checking'} {len(repo_ids)} repo(s) of {org} in {root / org}")
    each_repo(run, repo_ids, args.parallel_repos, download=False)
    saver.save()
    pending = [r for r in repo_ids if results[r].pending] if not (args.dry_run or args.no_download) else []
    if pending:
        _log(f"Downloading new and changed files of {len(pending)} repo(s)")
        done["n"] = 0
        each_repo(run, pending, args.parallel_repos, download=True)
        saver.save()
    results_list = [results[r] for r in repo_ids]

    gone = []
    if args.scan_local and not (args.only or args.smallest) and (root / org).is_dir():
        names = {r.split("/", 1)[1] for r in listed}
        gone = sorted(p.name for p in (root / org).iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in names)

    failed = [r for r in results_list if r.error or r.failed]
    report = {
        "org": org,
        "data_dir": str(root),
        "dry_run": args.dry_run,
        "synced_at": now_timestamps(),
        "datasets_synced": len(results) - len(failed),
        "datasets_failed": len(failed),
        "files_downloaded": sum(r.downloaded for r in results_list),
        "files_to_download": sum(r.pending for r in results_list),
        "files_adopted": sum(r.adopted for r in results_list),
        "files_unchanged": sum(r.unchanged for r in results_list),
        "files_removed_on_hub": sum(len(r.removed) for r in results_list),
        "repos_not_on_hub": gone,
        "datasets": [
            {
                "repo_id": r.repo_id,
                "seconds": round(r.seconds, 1),
                "downloaded": r.downloaded,
                **({"to_download": r.pending} if r.pending else {}),
                "adopted": r.adopted,
                "unchanged": r.unchanged,
                "removed_on_hub": r.removed,
                **({"untracked": r.untracked} if args.scan_local else {}),
                **({"failed": r.failed} if r.failed else {}),
                **({"error": r.error} if r.error else {}),
            }
            for r in results_list
        ],
    }
    if not args.dry_run:
        report_path = state_path.parent / REPORT_NAME
        tmp = report_path.with_name(f"{REPORT_NAME}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, report_path)
    _log(
        f"Downloaded {report['files_downloaded']} file(s), {report['files_to_download']} to download, adopted "
        f"{report['files_adopted']}, unchanged {report['files_unchanged']}, removed on Hub "
        f"{report['files_removed_on_hub']}, repos failed {len(failed)}"
    )
    if gone:
        _log(f"repo folders the Hub no longer has: {', '.join(gone)}")
    for r in failed:
        _log(f"failed: {r.repo_id}: {r.error or f'{len(r.failed)} file(s)'}", level=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
