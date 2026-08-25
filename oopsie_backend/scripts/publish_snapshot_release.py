"""Build and publish a frozen, privacy-processed Oopsie dataset snapshot.

The release builder never materializes unchanged source videos locally. It creates a
deterministic action plan whose unchanged videos are copied between Hub repositories
at pinned revisions, while locally rendered privacy replacements are uploaded from
their validated paths. The publisher is resumable and verifies every remote video
before creating the immutable release tag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import HfHubHTTPError


METADATA_PATHS = (
    "README.md",
    "release.json",
    "manifest.csv",
    "manifest.parquet",
    "privacy/privacy_decisions.parquet",
    "privacy/face_blur_manifest.csv",
)

MANIFEST_FIELDS = (
    "release_path",
    "source_repo",
    "source_revision",
    "source_path",
    "episode_key",
    "video_id",
    "source_file_size_bytes",
    "source_file_sha256",
    "release_size_bytes",
    "release_sha256",
    "face_blurred",
    "blur_exemption",
    "privacy_status",
    "privacy_decision_source",
    "publication_action",
    "release_version",
)

PRIVACY_FIELDS = (
    "release_path",
    "source_repo",
    "source_revision",
    "source_path",
    "episode_key",
    "video_id",
    "deface_status",
    "deface_detected",
    "deface_score_threshold",
    "deface_max_score",
    "deface_positive_frames",
    "deface_detection_count",
    "qwen_evaluated",
    "qwen_verified_face",
    "qwen_verification_status",
    "qwen_positive_frames",
    "qwen_negative_frames",
    "annotated_face_present",
    "annotated_blur_action",
    "annotation_source",
    "face_blurred",
    "blur_exemption",
    "privacy_status",
    "privacy_decision_source",
)

FACE_BLUR_FIELDS = (
    "release_path",
    "source_repo",
    "release_source_revision",
    "source_path",
    "video_id",
    "blur_source",
    "privacy_decision_source",
    "decision_date",
    "original_source_revision",
    "original_size_bytes",
    "original_sha256",
    "blurred_size_bytes",
    "blurred_sha256",
    "masked_frame_count",
    "region_count",
    "track_count",
    "confirmed_track_count",
    "fallback_detection_count",
    "config_hash",
)

MANIFEST_SCHEMA = pa.schema(
    [
        ("release_path", pa.string()),
        ("source_repo", pa.string()),
        ("source_revision", pa.string()),
        ("source_path", pa.string()),
        ("episode_key", pa.string()),
        ("video_id", pa.string()),
        ("source_file_size_bytes", pa.int64()),
        ("source_file_sha256", pa.string()),
        ("release_size_bytes", pa.int64()),
        ("release_sha256", pa.string()),
        ("face_blurred", pa.bool_()),
        ("blur_exemption", pa.string()),
        ("privacy_status", pa.string()),
        ("privacy_decision_source", pa.string()),
        ("publication_action", pa.string()),
        ("release_version", pa.string()),
    ]
)

PRIVACY_SCHEMA = pa.schema(
    [
        ("release_path", pa.string()),
        ("source_repo", pa.string()),
        ("source_revision", pa.string()),
        ("source_path", pa.string()),
        ("episode_key", pa.string()),
        ("video_id", pa.string()),
        ("deface_status", pa.string()),
        ("deface_detected", pa.bool_()),
        ("deface_score_threshold", pa.float64()),
        ("deface_max_score", pa.float64()),
        ("deface_positive_frames", pa.int64()),
        ("deface_detection_count", pa.int64()),
        ("qwen_evaluated", pa.bool_()),
        ("qwen_verified_face", pa.bool_()),
        ("qwen_verification_status", pa.string()),
        ("qwen_positive_frames", pa.int64()),
        ("qwen_negative_frames", pa.int64()),
        ("annotated_face_present", pa.string()),
        ("annotated_blur_action", pa.string()),
        ("annotation_source", pa.string()),
        ("face_blurred", pa.bool_()),
        ("blur_exemption", pa.string()),
        ("privacy_status", pa.string()),
        ("privacy_decision_source", pa.string()),
    ]
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _atomic_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]
) -> None:
    rows = list(rows)
    fields = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_hub_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe Hugging Face path: {value!r}")
    return path.as_posix()


def _key(repo: str, path: str) -> tuple[str, str]:
    return repo.strip(), _safe_hub_path(path.strip())


def _optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _index_unique(
    rows: Iterable[dict[str, str]], repo_field: str, path_field: str, label: str
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_key = _key(row[repo_field], row[path_field])
        if row_key in output:
            raise ValueError(f"duplicate {label} row: {row_key}")
        output[row_key] = row
    return output


def _load_annotations(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in _load_csv(path):
            row_key = _key(row["source_repo"], row["source_path"])
            if row_key in output:
                raise ValueError(f"duplicate annotation across inputs: {row_key}")
            output[row_key] = row
    return output


def _load_overlays(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in _load_csv(path):
            row_key = _key(row["source_repo"], row["source_path"])
            if row_key in output:
                raise ValueError(f"duplicate blurred overlay across inputs: {row_key}")
            output[row_key] = row
    return output


def _fetch_source_info(
    api: HfApi,
    inventory_rows: list[dict[str, str]],
    *,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in inventory_rows:
        grouped[(row["repo_id"], row["repo_revision"])].append(row["source_path"])
    jobs: list[tuple[str, str, list[str]]] = []
    for (repo, revision), paths in sorted(grouped.items()):
        for start in range(0, len(paths), 500):
            jobs.append((repo, revision, paths[start : start + 500]))

    def fetch(job: tuple[str, str, list[str]]) -> list[tuple[tuple[str, str], dict]]:
        repo, revision, paths = job
        infos = api.get_paths_info(
            repo,
            paths,
            repo_type="dataset",
            revision=revision,
            expand=False,
        )
        result = []
        for info in infos:
            lfs = getattr(info, "lfs", None)
            if lfs is None or not getattr(lfs, "sha256", None):
                raise ValueError(
                    f"video is not Hub LFS/Xet-backed; cannot server-copy safely: "
                    f"{repo}/{info.path}"
                )
            result.append(
                (
                    _key(repo, info.path),
                    {"size": int(info.size), "sha256": str(lfs.sha256)},
                )
            )
        return result

    output: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            for row_key, info in future.result():
                if row_key in output:
                    raise ValueError(f"duplicate remote source info: {row_key}")
                output[row_key] = info
            if index % 10 == 0 or index == len(futures):
                print(f"resolved source metadata batch {index}/{len(futures)}", flush=True)
    return output


def _load_carried_blur_rows(
    api: HfApi,
    carried_keys: set[tuple[str, str]],
    inventory: dict[tuple[str, str], dict[str, str]],
    metadata_path: str,
) -> dict[tuple[str, str], dict[str, str]]:
    repos = sorted({repo for repo, _ in carried_keys})
    output: dict[tuple[str, str], dict[str, str]] = {}
    for repo in repos:
        repo_keys = {key for key in carried_keys if key[0] == repo}
        revisions = {inventory[key]["repo_revision"] for key in repo_keys}
        if len(revisions) != 1:
            raise ValueError(f"carried blur repository spans revisions: {repo}")
        revision = next(iter(revisions))
        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                filename=metadata_path,
                token=api.token,
            )
        )
        rows = _index_unique(
            _load_csv(downloaded), "source_repo", "source_path", metadata_path
        )
        missing = repo_keys - set(rows)
        if missing:
            raise ValueError(
                f"{repo} final blur metadata is missing {len(missing)} carried videos"
            )
        for row_key in repo_keys:
            output[row_key] = rows[row_key]
    return output


def _privacy_status(
    *,
    annotation: dict[str, str] | None,
    deface: dict[str, str],
    qwen: dict[str, str] | None,
    blurred: bool,
) -> tuple[str, str, str | None]:
    exemption = (annotation or {}).get("exemption_reason", "").strip() or None
    if blurred:
        if annotation is None:
            raise ValueError("blurred video has no release annotation")
        return "blurred", annotation["annotation_source"], None
    if exemption:
        if annotation is None or annotation.get("face_present") != "yes":
            raise ValueError("blur exemption lacks a positive face annotation")
        return f"exempt_{exemption}", annotation["annotation_source"], exemption
    if annotation is not None and annotation.get("face_present") == "no":
        return "manual_no_face", annotation["annotation_source"], None
    if deface["status"] != "ok":
        return "scan_error", "deface_scan_2026-08-22", None
    if not _optional_bool(deface["has_face"]):
        return "deface_not_detected", "deface_scan_2026-08-22", None
    if qwen is None:
        raise ValueError("Deface-positive video is missing Qwen verification")
    if _optional_bool(qwen["qwen_verified_face"]):
        raise ValueError(
            "Qwen-positive video has neither blur output nor a prior annotation"
        )
    return "qwen_rejected", "qwen_verification_2026-08-23", None


def _readme(release: dict[str, Any]) -> str:
    repo_lines = "\n".join(
        f"| `{row['repo_id'].rsplit('/', 1)[-1]}` | {row['video_count']:,} | "
        f"`{row['revision']}` |"
        for row in release["source_repositories"]
    )
    return f"""---
pretty_name: OopsieData v0.1
---

# OopsieData v0.1

OopsieData v0.1 is a consolidated, privacy-processed release containing
{release['totals']['videos']:,} videos from
{release['totals']['repositories']} contributing repositories. Videos retain
their original repository-relative paths under a lab-organized layout.

Videos selected by the privacy pipeline are provided as face-blurred renders.

## Layout

```text
labs/<source-repository>/<original-source-path>
manifest.parquet
manifest.csv
privacy/privacy_decisions.parquet
privacy/face_blur_manifest.csv
release.json
```

`manifest.parquet` is the canonical one-row-per-video release manifest. The CSV
contains the same records for convenience. `release.json` records release
metadata and the pinned source revisions.

## Download

Download the versioned release:

```bash
hf download {release['repo_id']} \\
  --repo-type dataset \\
  --revision v0.1 \\
  --local-dir ./oopsiedata-v0.1
```

Use `snapshot_download(..., allow_patterns=["labs/UofT_TISL/**"])` to download
only one lab repository.

## Privacy processing

- Face-blurred videos: {release['totals']['face_blurred_videos']:,}
- Preserved face-containing exemptions: {release['totals']['blur_exemptions']:,}

`privacy/privacy_decisions.parquet` explains the privacy outcome for every
release video. `privacy/face_blur_manifest.csv` lists only videos whose release
bytes are face-blurred, together with final hashes and available render audit
statistics.

## Source repositories

| Repository | Videos | Revision |
|---|---:|---|
{repo_lines}
"""


def prepare_release(args: argparse.Namespace) -> None:
    scan_dir = args.scan_dir.resolve()
    output_dir = args.output_dir.resolve()
    internal_dir = output_dir / ".internal"
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)

    scope_path = scan_dir / "scope" / "scope.json"
    inventory_path = scan_dir / "scope" / "inventory.csv"
    deface_path = scan_dir / "deface-0.4" / "videos.csv"
    qwen_path = scan_dir / "qwen" / "videos.csv"
    scope = _load_json(scope_path)
    inventory_rows = _load_csv(inventory_path)
    inventory = _index_unique(inventory_rows, "repo_id", "source_path", "inventory")
    deface = _index_unique(
        _load_csv(deface_path), "source_repo", "source_path", "Deface scan"
    )
    qwen = _index_unique(
        _load_csv(qwen_path), "source_repo", "source_path", "Qwen verification"
    )
    annotations = _load_annotations(args.annotation_manifest)
    overlays = _load_overlays(args.overlay_blur_manifest)

    if len(inventory) != scope["videos"]:
        raise ValueError("scope and inventory video counts differ")
    if set(deface) != set(inventory):
        raise ValueError("Deface scan does not cover the exact frozen inventory")
    expected_qwen = {
        key
        for key, row in deface.items()
        if row["status"] == "ok" and _optional_bool(row["has_face"])
    }
    if set(qwen) != expected_qwen:
        raise ValueError("Qwen results do not cover the exact Deface-positive set")

    ignored_annotations = sorted(set(annotations) - set(inventory))
    in_scope_annotations = {
        key: row for key, row in annotations.items() if key in inventory
    }
    blur_keys = {
        key
        for key, row in in_scope_annotations.items()
        if row["blur_action"] == "blur"
    }
    if set(overlays) - set(inventory):
        raise ValueError("a local blurred overlay is outside the frozen inventory")
    if set(overlays) - blur_keys:
        raise ValueError("a local blurred overlay lacks a blur annotation")
    carried_keys = blur_keys - set(overlays)
    if len(overlays) != 154 or len(carried_keys) != 240 or len(blur_keys) != 394:
        raise ValueError(
            "unexpected privacy release counts: "
            f"overlays={len(overlays)}, carried={len(carried_keys)}, "
            f"blurred={len(blur_keys)}"
        )

    api = HfApi()
    source_info = _fetch_source_info(api, inventory_rows, workers=args.api_workers)
    if set(source_info) != set(inventory):
        missing = set(inventory) - set(source_info)
        raise ValueError(f"missing Hub source metadata for {len(missing)} videos")
    carried = _load_carried_blur_rows(
        api,
        carried_keys,
        inventory,
        args.carried_blur_metadata_path,
    )

    for row_key, row in overlays.items():
        local_path = Path(row["output_local_path"])
        if not local_path.is_file():
            raise FileNotFoundError(f"missing blurred overlay: {local_path}")
        if local_path.stat().st_size != int(row["blurred_size_bytes"]):
            raise ValueError(f"blurred overlay size changed: {row_key}")
        if _sha256(local_path) != row["blurred_sha256"]:
            raise ValueError(f"blurred overlay hash changed: {row_key}")
        info = source_info[row_key]
        if (
            info["size"] != int(row["source_size_bytes"])
            or info["sha256"] != row["source_sha256"]
        ):
            raise ValueError(f"overlay source differs from frozen revision: {row_key}")

    for row_key, row in carried.items():
        info = source_info[row_key]
        if (
            info["size"] != int(row["blurred_size_bytes"])
            or info["sha256"] != row["blurred_sha256"]
        ):
            raise ValueError(
                f"carried blurred bytes differ from source metadata: {row_key}"
            )

    manifest_rows: list[dict[str, Any]] = []
    privacy_rows: list[dict[str, Any]] = []
    blur_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    repo_bytes: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()

    for inventory_row in sorted(
        inventory_rows, key=lambda row: (row["repo_id"], row["source_path"])
    ):
        row_key = _key(inventory_row["repo_id"], inventory_row["source_path"])
        annotation = in_scope_annotations.get(row_key)
        deface_row = deface[row_key]
        qwen_row = qwen.get(row_key)
        blurred = row_key in blur_keys
        privacy_status, decision_source, exemption = _privacy_status(
            annotation=annotation,
            deface=deface_row,
            qwen=qwen_row,
            blurred=blurred,
        )
        short_repo = row_key[0].rsplit("/", 1)[-1]
        release_path = _safe_hub_path(f"labs/{short_repo}/{row_key[1]}")
        folder_counts[str(PurePosixPath(release_path).parent)] += 1
        source = source_info[row_key]
        if row_key in overlays:
            overlay = overlays[row_key]
            release_size = int(overlay["blurred_size_bytes"])
            release_sha = overlay["blurred_sha256"]
            action = "local_upload"
            local_path = str(Path(overlay["output_local_path"]).resolve())
        else:
            release_size = source["size"]
            release_sha = source["sha256"]
            action = "server_copy"
            local_path = None

        manifest_row = {
            "release_path": release_path,
            "source_repo": row_key[0],
            "source_revision": inventory_row["repo_revision"],
            "source_path": row_key[1],
            "episode_key": inventory_row["episode_key"],
            "video_id": inventory_row["video_id"],
            "source_file_size_bytes": source["size"],
            "source_file_sha256": source["sha256"],
            "release_size_bytes": release_size,
            "release_sha256": release_sha,
            "face_blurred": blurred,
            "blur_exemption": exemption,
            "privacy_status": privacy_status,
            "privacy_decision_source": decision_source,
            "publication_action": action,
            "release_version": args.release_version,
        }
        manifest_rows.append(manifest_row)
        privacy_rows.append(
            {
                "release_path": release_path,
                "source_repo": row_key[0],
                "source_revision": inventory_row["repo_revision"],
                "source_path": row_key[1],
                "episode_key": inventory_row["episode_key"],
                "video_id": inventory_row["video_id"],
                "deface_status": deface_row["status"],
                "deface_detected": _optional_bool(deface_row["has_face"]),
                "deface_score_threshold": _optional_float(
                    deface_row["score_threshold"]
                ),
                "deface_max_score": _optional_float(deface_row["max_score"]),
                "deface_positive_frames": _optional_int(
                    deface_row["positive_frame_count"]
                ),
                "deface_detection_count": _optional_int(
                    deface_row["detection_count"]
                ),
                "qwen_evaluated": qwen_row is not None,
                "qwen_verified_face": (
                    _optional_bool(qwen_row["qwen_verified_face"])
                    if qwen_row is not None
                    else None
                ),
                "qwen_verification_status": (
                    qwen_row["verification_status"] if qwen_row is not None else None
                ),
                "qwen_positive_frames": (
                    _optional_int(qwen_row["qwen_positive_frames"])
                    if qwen_row is not None
                    else None
                ),
                "qwen_negative_frames": (
                    _optional_int(qwen_row["qwen_negative_frames"])
                    if qwen_row is not None
                    else None
                ),
                "annotated_face_present": (
                    annotation.get("face_present") if annotation else None
                ),
                "annotated_blur_action": (
                    annotation.get("blur_action") if annotation else None
                ),
                "annotation_source": (
                    annotation.get("annotation_source") if annotation else None
                ),
                "face_blurred": blurred,
                "blur_exemption": exemption,
                "privacy_status": privacy_status,
                "privacy_decision_source": decision_source,
            }
        )
        actions.append(
            {
                "action": action,
                "release_path": release_path,
                "source_repo": row_key[0],
                "source_revision": inventory_row["repo_revision"],
                "source_path": row_key[1],
                "local_path": local_path,
                "expected_size_bytes": release_size,
                "expected_sha256": release_sha,
            }
        )
        if blurred:
            audit = overlays.get(row_key) or carried[row_key]
            blur_rows.append(
                {
                    "release_path": release_path,
                    "source_repo": row_key[0],
                    "release_source_revision": inventory_row["repo_revision"],
                    "source_path": row_key[1],
                    "video_id": inventory_row["video_id"],
                    "blur_source": (
                        "newly_rendered_for_v0.1"
                        if row_key in overlays
                        else "carried_forward_from_source_repo"
                    ),
                    "privacy_decision_source": decision_source,
                    "decision_date": (annotation or {}).get("annotated_at") or None,
                    "original_source_revision": audit.get("source_revision") or None,
                    "original_size_bytes": _optional_int(
                        audit.get("source_size_bytes")
                    ),
                    "original_sha256": audit.get("source_sha256") or None,
                    "blurred_size_bytes": release_size,
                    "blurred_sha256": release_sha,
                    "masked_frame_count": _optional_int(
                        audit.get("masked_frame_count")
                    ),
                    "region_count": _optional_int(audit.get("region_count")),
                    "track_count": _optional_int(audit.get("track_count")),
                    "confirmed_track_count": _optional_int(
                        audit.get("confirmed_track_count")
                    ),
                    "fallback_detection_count": _optional_int(
                        audit.get("fallback_detection_count")
                    ),
                    "config_hash": audit.get("config_hash") or None,
                }
            )
        status_counts[privacy_status] += 1
        repo_counts[row_key[0]] += 1
        repo_bytes[row_key[0]] += release_size

    if max(folder_counts.values()) >= 10_000:
        raise ValueError("release layout exceeds the Hub per-folder file limit")
    if len({row["release_path"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("release paths are not unique")

    source_repositories = [
        {
            "repo_id": repo,
            "revision": next(
                iter(
                    {
                        row["repo_revision"]
                        for row in inventory_rows
                        if row["repo_id"] == repo
                    }
                )
            ),
            "video_count": repo_counts[repo],
            "release_bytes": repo_bytes[repo],
        }
        for repo in sorted(repo_counts)
    ]
    exempt_count = sum(1 for row in manifest_rows if row["blur_exemption"])
    release = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "release_version": args.release_version,
        "visibility": "private",
        "generated_at": _utc_now(),
        "frozen_at": scope["frozen_at"],
        "release_scope": "frozen video inventory plus privacy-processed overlays",
        "totals": {
            "repositories": len(source_repositories),
            "videos": len(manifest_rows),
            "release_bytes": sum(row["release_size_bytes"] for row in manifest_rows),
            "server_copied_videos": sum(
                row["publication_action"] == "server_copy" for row in manifest_rows
            ),
            "overlay_videos": len(overlays),
            "face_blurred_videos": len(blur_rows),
            "carried_blurred_videos": len(carried),
            "blur_exemptions": exempt_count,
            "ignored_out_of_scope_annotations": len(ignored_annotations),
        },
        "privacy_status_counts": dict(sorted(status_counts.items())),
        "source_repositories": source_repositories,
        "excluded_repositories": sorted(
            scope["public_excluded_datasets"] + scope["non_source_datasets"]
        ),
        "privacy_pipeline": {
            "deface": scope["deface"],
            "qwen": scope["qwen"],
        },
        "input_artifacts": {
            "inventory_sha256": _sha256(inventory_path),
            "deface_videos_sha256": _sha256(deface_path),
            "qwen_videos_sha256": _sha256(qwen_path),
            "annotation_manifests": [
                {"name": path.name, "sha256": _sha256(path)}
                for path in args.annotation_manifest
            ],
            "overlay_blur_manifests": [
                {"name": path.name, "sha256": _sha256(path)}
                for path in args.overlay_blur_manifest
            ],
        },
        "metadata": {
            "manifest": "manifest.parquet",
            "manifest_csv": "manifest.csv",
            "privacy_decisions": "privacy/privacy_decisions.parquet",
            "face_blur_manifest": "privacy/face_blur_manifest.csv",
        },
    }

    _atomic_csv(output_dir / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    _atomic_parquet(output_dir / "manifest.parquet", manifest_rows, MANIFEST_SCHEMA)
    _atomic_parquet(
        output_dir / "privacy" / "privacy_decisions.parquet",
        privacy_rows,
        PRIVACY_SCHEMA,
    )
    _atomic_csv(
        output_dir / "privacy" / "face_blur_manifest.csv",
        blur_rows,
        FACE_BLUR_FIELDS,
    )
    _atomic_json(output_dir / "release.json", release)
    (output_dir / "README.md").write_text(_readme(release), encoding="utf-8")
    actions_path = internal_dir / "publish_actions.jsonl"
    with actions_path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, sort_keys=True) + "\n")
    preparation = {
        "schema_version": 1,
        "prepared_at": _utc_now(),
        "release_plan_sha256": _sha256(actions_path),
        "metadata_sha256": {
            path: _sha256(output_dir / path) for path in METADATA_PATHS
        },
        "ignored_annotations": [
            {"source_repo": repo, "source_path": path}
            for repo, path in ignored_annotations
        ],
    }
    _atomic_json(internal_dir / "preparation.json", preparation)
    print(json.dumps(release, indent=2, sort_keys=True), flush=True)


def _load_actions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"invalid release action in {path}")
                rows.append(value)
    return rows


def _remote_matches(
    api: HfApi,
    *,
    repo_id: str,
    actions: list[dict[str, Any]],
    revision: str = "main",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    infos = api.get_paths_info(
        repo_id,
        [row["release_path"] for row in actions],
        repo_type="dataset",
        revision=revision,
        expand=False,
    )
    by_path = {info.path: info for info in infos}
    present = []
    missing = []
    for row in actions:
        info = by_path.get(row["release_path"])
        if info is None:
            missing.append(row)
            continue
        lfs = getattr(info, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None)
        if int(info.size) != int(row["expected_size_bytes"]) or remote_sha != row[
            "expected_sha256"
        ]:
            raise ValueError(
                f"remote path has unexpected bytes: {repo_id}/{row['release_path']}"
            )
        present.append(row)
    return present, missing


def _metadata_is_current(api: HfApi, repo_id: str, release_dir: Path) -> bool:
    try:
        infos = api.get_paths_info(
            repo_id,
            list(METADATA_PATHS),
            repo_type="dataset",
            revision="main",
            expand=False,
        )
    except Exception:
        return False
    if len(infos) != len(METADATA_PATHS):
        return False
    for info in infos:
        local = release_dir / info.path
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            if lfs.sha256 != _sha256(local):
                return False
        else:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision="main",
                    filename=info.path,
                    token=api.token,
                )
            )
            if _sha256(downloaded) != _sha256(local):
                return False
    return True


def _batch_actions(
    actions: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        group = (
            row["action"],
            row["source_repo"] if row["action"] == "server_copy" else "local",
        )
        grouped[group].append(row)
    batches = []
    for group in sorted(grouped):
        rows = sorted(grouped[group], key=lambda row: row["release_path"])
        for start in range(0, len(rows), batch_size):
            batches.append(rows[start : start + batch_size])
    return batches


def _save_publication(path: Path, publication: dict[str, Any]) -> None:
    publication["updated_at"] = _utc_now()
    _atomic_json(path, publication)


def _hub_write_with_rate_limit_retry(description: str, operation):
    """Retry Hub writes after either the general or commit-specific 429 window."""
    attempts = 0
    while True:
        try:
            return operation()
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            if response is None or response.status_code != 429:
                raise
            attempts += 1
            if attempts > 12:
                raise
            message = str(exc).lower()
            retry_header = response.headers.get("retry-after", "")
            try:
                wait_seconds = int(float(retry_header)) + 10
            except ValueError:
                wait_seconds = 310
            if "repository commits" in message or "commits (320 per hour)" in message:
                wait_seconds = max(wait_seconds, 3_700)
            print(
                f"Hub rate limit reached during {description}; retrying in "
                f"{wait_seconds} seconds (attempt {attempts}/12)",
                flush=True,
            )
            time.sleep(wait_seconds)


def publish_release(args: argparse.Namespace) -> None:
    release_dir = args.release_dir.resolve()
    release = _load_json(release_dir / "release.json")
    preparation = _load_json(release_dir / ".internal" / "preparation.json")
    actions_path = release_dir / ".internal" / "publish_actions.jsonl"
    if _sha256(actions_path) != preparation["release_plan_sha256"]:
        raise ValueError("release action plan changed after preparation")
    for path, expected in preparation["metadata_sha256"].items():
        if _sha256(release_dir / path) != expected:
            raise ValueError(f"release metadata changed after preparation: {path}")
    if release["repo_id"] != args.repo_id:
        raise ValueError("prepared destination differs from requested repository")
    if release["visibility"] != "private":
        raise ValueError("this publisher requires a private prepared release")

    api = HfApi()
    identity = api.whoami().get("name")
    if identity != args.expected_identity:
        raise PermissionError(
            f"refusing Hub writes as {identity!r}; expected {args.expected_identity!r}"
        )
    api.create_repo(
        args.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if not info.private:
        raise PermissionError(f"refusing to publish because {args.repo_id} is not private")

    state_path = release_dir / ".internal" / "publication.json"
    publication = (
        _load_json(state_path)
        if state_path.is_file()
        else {
            "schema_version": 1,
            "repo_id": args.repo_id,
            "release_version": release["release_version"],
            "started_at": _utc_now(),
            "metadata_commit": None,
            "completed_batches": {},
            "remote_validation": None,
            "tagged": False,
        }
    )
    if publication["repo_id"] != args.repo_id:
        raise ValueError("publication state targets a different repository")

    if not _metadata_is_current(api, args.repo_id, release_dir):
        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=release_dir / path)
            for path in METADATA_PATHS
        ]
        commit = _hub_write_with_rate_limit_retry(
            "release metadata commit",
            lambda: api.create_commit(
                args.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message="Add OopsieData v0.1 release metadata",
                commit_description=(
                    "Initialize the private snapshot with its frozen source inventory, "
                    "privacy decisions, and publication manifest."
                ),
                num_threads=args.upload_threads,
            ),
        )
        publication["metadata_commit"] = commit.oid
        _save_publication(state_path, publication)
        print(f"published release metadata: {commit.oid}", flush=True)
    else:
        print("release metadata already matches", flush=True)

    actions = _load_actions(actions_path)
    if len(actions) != release["totals"]["videos"]:
        raise ValueError("release action count differs from release metadata")
    batches = _batch_actions(actions, args.batch_size)
    for index, batch in enumerate(batches, 1):
        batch_id = _stable_hash(
            [
                {
                    "action": row["action"],
                    "release_path": row["release_path"],
                    "expected_sha256": row["expected_sha256"],
                }
                for row in batch
            ]
        )
        previous = publication["completed_batches"].get(batch_id)
        if previous:
            print(f"batch {index}/{len(batches)} already complete", flush=True)
            continue
        present, missing = _remote_matches(
            api, repo_id=args.repo_id, actions=batch, revision="main"
        )
        if not missing:
            publication["completed_batches"][batch_id] = {
                "action": batch[0]["action"],
                "video_count": len(batch),
                "reconciled": True,
                "completed_at": _utc_now(),
            }
            _save_publication(state_path, publication)
            print(
                f"batch {index}/{len(batches)} reconciled from remote state",
                flush=True,
            )
            continue

        operations = []
        for row in missing:
            if row["action"] == "server_copy":
                operations.append(
                    CommitOperationCopy(
                        src_path_in_repo=row["source_path"],
                        path_in_repo=row["release_path"],
                        src_revision=row["source_revision"],
                        src_repo_id=row["source_repo"],
                        src_repo_type="dataset",
                    )
                )
            elif row["action"] == "local_upload":
                local_path = Path(row["local_path"])
                if (
                    not local_path.is_file()
                    or local_path.stat().st_size != int(row["expected_size_bytes"])
                    or _sha256(local_path) != row["expected_sha256"]
                ):
                    raise ValueError(f"local overlay changed: {local_path}")
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=row["release_path"], path_or_fileobj=local_path
                    )
                )
            else:
                raise ValueError(f"unknown publication action: {row['action']}")
        head = api.repo_info(args.repo_id, repo_type="dataset", revision="main").sha
        commit = _hub_write_with_rate_limit_retry(
            f"video batch {index}/{len(batches)}",
            lambda: api.create_commit(
                args.repo_id,
                repo_type="dataset",
                revision="main",
                parent_commit=head,
                operations=operations,
                commit_message=(
                    f"Publish {len(missing)} v0.1 videos "
                    f"({batch[0]['action']}, batch {index}/{len(batches)})"
                ),
                num_threads=args.upload_threads,
            ),
        )
        _, still_missing = _remote_matches(
            api, repo_id=args.repo_id, actions=batch, revision=commit.oid
        )
        if still_missing:
            raise ValueError(f"remote commit is missing {len(still_missing)} batch files")
        publication["completed_batches"][batch_id] = {
            "action": batch[0]["action"],
            "video_count": len(batch),
            "already_present": len(present),
            "published": len(missing),
            "commit": commit.oid,
            "completed_at": _utc_now(),
        }
        _save_publication(state_path, publication)
        print(
            f"published batch {index}/{len(batches)}: {len(missing)} videos, "
            f"commit {commit.oid}",
            flush=True,
        )

    print("validating all release videos on the Hub", flush=True)
    for start in range(0, len(actions), 500):
        batch = actions[start : start + 500]
        _, missing = _remote_matches(
            api, repo_id=args.repo_id, actions=batch, revision="main"
        )
        if missing:
            raise ValueError(f"remote release is missing {len(missing)} videos")
        print(
            f"validated {min(start + len(batch), len(actions))}/{len(actions)} videos",
            flush=True,
        )
    head = api.repo_info(args.repo_id, repo_type="dataset", revision="main").sha
    publication["remote_validation"] = {
        "validated_at": _utc_now(),
        "revision": head,
        "video_count": len(actions),
        "release_bytes": release["totals"]["release_bytes"],
        "release_plan_sha256": preparation["release_plan_sha256"],
    }
    _save_publication(state_path, publication)
    _hub_write_with_rate_limit_retry(
        "release tag creation",
        lambda: api.create_tag(
            args.repo_id,
            repo_type="dataset",
            tag=release["release_version"],
            revision=head,
            tag_message="Immutable OopsieData v0.1 privacy-processed snapshot",
            exist_ok=True,
        ),
    )
    tag_head = api.repo_info(
        args.repo_id, repo_type="dataset", revision=release["release_version"]
    ).sha
    if tag_head != head:
        raise ValueError(
            f"release tag points to {tag_head}, expected validated commit {head}"
        )
    publication["tagged"] = True
    publication["tag"] = release["release_version"]
    publication["tag_revision"] = tag_head
    publication["completed_at"] = _utc_now()
    _save_publication(state_path, publication)
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build and validate metadata")
    prepare.add_argument("--scan-dir", type=Path, required=True)
    prepare.add_argument(
        "--annotation-manifest", type=Path, action="append", required=True
    )
    prepare.add_argument(
        "--overlay-blur-manifest", type=Path, action="append", required=True
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--repo-id", required=True)
    prepare.add_argument("--release-version", default="v0.1")
    prepare.add_argument(
        "--carried-blur-metadata-path",
        default="face_blur_release/2026-08-12/manifest.csv",
    )
    prepare.add_argument("--api-workers", type=int, default=6)
    prepare.set_defaults(func=prepare_release)

    publish = subparsers.add_parser(
        "publish", help="create the private repo, publish, verify, and tag"
    )
    publish.add_argument("--release-dir", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--expected-identity", default="oopsiedata")
    publish.add_argument("--batch-size", type=int, default=75)
    publish.add_argument("--upload-threads", type=int, default=8)
    publish.set_defaults(func=publish_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
