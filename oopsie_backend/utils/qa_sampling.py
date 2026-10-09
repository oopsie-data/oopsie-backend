"""Select diverse episodes per lab for QA spot checks and bundle them into zips."""

from __future__ import annotations

import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import h5py

from oopsie_backend.utils.episode_stats import (
    _include_in_public_stats,
    _s,
    extract_record,
    now_timestamps,
)

NO_ANNOTATOR = "<none>"
DIMENSIONS = ("language_instruction", "annotator", "operator", "robot_name")
MANIFEST_NAME = "qa_manifest.json"


def _video_files(h5_path: Path, repo_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map camera -> ``(source_file, repo_relative_bundle_path)``.

    The bundle path is where the annotator will look first for the stored
    ``video_paths`` entry, so the copied h5 resolves its videos unchanged.
    Falls back to ``<stem>_<cam>.mp4`` next to the h5.
    """
    repo_dir = repo_dir.resolve()
    h5_dir = h5_path.parent.resolve()
    out: dict[str, tuple[Path, str]] = {}
    with h5py.File(h5_path, "r") as f:
        group = f.get("observations/video_paths")
        raw_paths = (
            {str(cam): _s(group[cam][()]) for cam in group}
            if isinstance(group, h5py.Group)
            else {}
        )
    for cam, raw in raw_paths.items():
        if not raw:
            continue
        candidate = PurePosixPath(raw)
        target = None
        if not candidate.is_absolute():
            target = (h5_dir / candidate).resolve()
            if not target.is_relative_to(repo_dir):
                target = None
        if target is None:
            target = h5_dir / candidate.name
        source = target if target.is_file() else h5_dir / candidate.name
        if not source.is_file():
            continue
        out[cam] = (source, target.relative_to(repo_dir).as_posix())
    if not out:
        stem = h5_path.stem
        for mp4 in sorted(h5_path.parent.glob(f"{stem}_*.mp4")):
            cam = mp4.stem[len(stem) + 1 :]
            if cam:
                out[cam] = (mp4, mp4.resolve().relative_to(repo_dir).as_posix())
    return out


def repo_root(path: Path, repo_path: str) -> Path:
    """The HF repo directory containing ``path``, whose path inside the repo is ``repo_path``."""
    root = path.resolve()
    for _ in PurePosixPath(repo_path).parts:
        root = root.parent
    return root


def episode_videos(path: Path, repo_path: str) -> dict[str, Path]:
    """Map camera -> mp4 file of the episode at ``path``."""
    return {cam: src for cam, (src, _) in _video_files(path, repo_root(path, repo_path)).items()}


def _load_one(args: tuple[str, str, Path, Path]) -> dict | str:
    dataset_repo, repo_path, path, repo_dir = args
    try:
        record = extract_record(path, repo_path)
        videos = _video_files(path, repo_dir)
    except Exception as exc:
        return f"{dataset_repo} :: {repo_path}: {exc}"
    if not videos:
        return f"{dataset_repo} :: {repo_path}: no video files found"
    record["dataset_repo"] = dataset_repo
    record["local_path"] = str(path)
    record["videos"] = {cam: (str(src), rel) for cam, (src, rel) in videos.items()}
    record["annotators"] = sorted({a["name"] for a in record["annotations"]})
    return record


def collect_records(
    episodes: list[tuple[str, str, Path]], data_dir: Path, workers: int = 8
) -> list[dict]:
    """Read every eligible episode's metadata. ``episodes`` comes from ``discover_episodes``."""
    data_dir = data_dir.resolve()
    jobs = []
    for dataset_repo, repo_path, path in episodes:
        if not _include_in_public_stats({"dataset_repo": dataset_repo}, dataset_repo):
            continue
        jobs.append((dataset_repo, repo_path, path, repo_root(path, repo_path)))

    records: list[dict] = []
    skipped: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load_one, jobs, chunksize=16):
            if isinstance(result, str):
                skipped.append(result)
            elif _include_in_public_stats(result, result["dataset_repo"]):
                records.append(result)
    for msg in skipped:
        print(f"  skipped {msg}", file=sys.stderr)
    print(
        f"collected {len(records)} episode(s), skipped {len(skipped)}", file=sys.stderr
    )
    return records


def _dimension_values(record: dict) -> dict[str, list[str]]:
    instruction = (record.get("language_instruction") or "").strip().lower()
    return {
        "language_instruction": [instruction],
        "annotator": record["annotators"] or [NO_ANNOTATOR],
        "operator": [record.get("operator_name") or ""],
        "robot_name": [record.get("robot_name") or ""],
    }


def select_diverse(records: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Greedily pick ``k`` records that cover as many dimension values as possible.

    Each candidate scores ``sum_d 1 / (1 + times its value of d was already
    picked)``; ties are broken randomly. Once every value has been picked the
    score degrades gracefully, so buckets are reused only when exhausted.
    """
    pool = list(records)
    rng.shuffle(pool)
    values = [_dimension_values(r) for r in pool]
    counts: dict[str, Counter] = {d: Counter() for d in DIMENSIONS}
    chosen: list[int] = []
    remaining = set(range(len(pool)))
    while remaining and len(chosen) < k:
        best = max(
            remaining,
            key=lambda i: (
                sum(
                    1.0 / (1 + min(counts[d][v] for v in values[i][d]))
                    for d in DIMENSIONS
                ),
                -i,
            ),
        )
        remaining.discard(best)
        chosen.append(best)
        for d in DIMENSIONS:
            counts[d].update(values[best][d])
    return [pool[i] for i in chosen]


def select_per_lab(
    records: list[dict], per_lab: int, seed: int, min_episodes: int = 0
) -> dict[str, list[dict]]:
    """Pick ``per_lab`` episodes from every lab with at least ``min_episodes`` episodes."""
    by_lab: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lab[r["dataset_repo"]].append(r)
    selection: dict[str, list[dict]] = {}
    for lab, rs in sorted(by_lab.items()):
        if len(rs) < min_episodes:
            print(
                f"  skipping {lab}: {len(rs)} episode(s) < {min_episodes}", file=sys.stderr
            )
            continue
        rs = sorted(rs, key=lambda r: r["repo_path"])
        selection[lab] = select_diverse(rs, per_lab, random.Random(f"{seed}:{lab}"))
    return selection


def diversity_report(
    all_records: list[dict], selection: dict[str, list[dict]]
) -> list[str]:
    """One line per lab: distinct values selected / available per dimension."""
    by_lab: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        by_lab[r["dataset_repo"]].append(r)

    def distinct(rs: list[dict], d: str) -> int:
        return len({v for r in rs for v in _dimension_values(r)[d]})

    lines = [
        f"{'lab':45s} {'n':>9s}  " + "  ".join(f"{d[:12]:>12s}" for d in DIMENSIONS)
    ]
    for lab, picked in selection.items():
        cols = "  ".join(
            f"{distinct(picked, d):>5d}/{distinct(by_lab[lab], d):<6d}" for d in DIMENSIONS
        )
        lines.append(f"{lab:45s} {len(picked):>4d}/{len(by_lab[lab]):<4d}  {cols}")
    return lines


def _manifest_entry(record: dict, qa_id: str) -> dict:
    repo_name = record["dataset_repo"].split("/", 1)[1]
    prefix = f"episodes/{repo_name}"
    return {
        "qa_id": qa_id,
        "dataset_repo": record["dataset_repo"],
        "lab_id": record.get("lab_id"),
        "repo_path": record["repo_path"],
        "bundle_path": f"{prefix}/{record['repo_path']}",
        "video_files": {cam: f"{prefix}/{rel}" for cam, (_, rel) in record["videos"].items()},
        "episode_id": record.get("episode_id"),
        "language_instruction": record.get("language_instruction"),
        "operator": record.get("operator_name"),
        "annotators": record["annotators"],
        "robot_name": record.get("robot_name"),
        "qa_notes": "",
    }


def write_bundles(
    selection: dict[str, list[dict]],
    out_dir: Path,
    labs_per_bundle: int,
    seed: int,
) -> list[dict]:
    """Write ``qa_bundle_XX.zip`` files and return their manifests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labs = sorted(selection)
    created_at = now_timestamps()["utc"]
    manifests: list[dict] = []
    for b, start in enumerate(range(0, len(labs), labs_per_bundle), 1):
        bundle_id = f"qa_bundle_{b:02d}"
        entries: list[dict] = []
        files: list[tuple[str, str]] = []
        for lab in labs[start : start + labs_per_bundle]:
            for record in selection[lab]:
                entry = _manifest_entry(record, f"{bundle_id}-{len(entries) + 1:03d}")
                entries.append(entry)
                files.append((record["local_path"], entry["bundle_path"]))
                for cam, (src, _) in record["videos"].items():
                    files.append((src, entry["video_files"][cam]))
        manifest = {
            "bundle_id": bundle_id,
            "created_at": created_at,
            "seed": seed,
            "reviewer": "",
            "labs": labs[start : start + labs_per_bundle],
            "episodes": entries,
        }
        zip_path = out_dir / f"{bundle_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(f"{bundle_id}/{MANIFEST_NAME}", json.dumps(manifest, indent=2))
            written: set[str] = set()
            for src, arc in files:
                if arc in written:
                    continue
                written.add(arc)
                zf.write(src, f"{bundle_id}/{arc}")
        print(f"wrote {zip_path} ({len(entries)} episodes)", file=sys.stderr)
        manifests.append(manifest)

    master = {"created_at": created_at, "seed": seed, "bundles": manifests}
    (out_dir / "qa_master_manifest.json").write_text(json.dumps(master, indent=2))
    return manifests
