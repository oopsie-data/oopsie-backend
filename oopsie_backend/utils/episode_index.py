"""Flat per-episode metadata index, written as JSONL, for offline split planning."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from oopsie_backend.utils.episode_stats import (
    _include_in_public_stats,
    _primary_sf_annotation,
    extract_record,
)

INDEX_FIELDS = (
    "dataset_repo",
    "repo_path",
    "episode_id",
    "lab_id",
    "robot_name",
    "policy_name",
    "operator_name",
    "language_instruction",
    "task",
    "cameras",
    "timestamp",
    "duration_seconds",
    "success",
    "has_failure_description",
    "annotators",
)


def normalize_task(instruction: str | None) -> str:
    """Lowercase, collapse whitespace and strip surrounding punctuation."""
    text = re.sub(r"\s+", " ", (instruction or "").strip().lower())
    return text.strip(" .!?,;:\"'")


def index_row(record: dict, dataset_repo: str) -> dict:
    primary = _primary_sf_annotation(record["annotations"])
    return {
        "dataset_repo": dataset_repo,
        "repo_path": record["repo_path"],
        "episode_id": record.get("episode_id") or "",
        "lab_id": record.get("lab_id"),
        "robot_name": record.get("robot_name"),
        "policy_name": record.get("policy_name"),
        "operator_name": record.get("operator_name") or "",
        "language_instruction": record.get("language_instruction") or "",
        "task": normalize_task(record.get("language_instruction")),
        "cameras": ",".join(record.get("cameras_present") or []),
        "timestamp": record.get("timestamp"),
        "duration_seconds": record.get("duration_seconds"),
        "success": primary["success"] if primary else None,
        "has_failure_description": bool(primary and primary["has_failure_description"]),
        "annotators": sorted({a["name"] for a in record["annotations"]}),
    }


def _load_one(args: tuple[str, str, Path]) -> dict | str:
    dataset_repo, repo_path, path = args
    try:
        record = extract_record(path, repo_path)
    except Exception as exc:
        return f"{dataset_repo} :: {repo_path}: {exc}"
    if not _include_in_public_stats(record, dataset_repo):
        return ""
    return index_row(record, dataset_repo)


def build_index(episodes: list[tuple[str, str, Path]], workers: int = 8) -> list[dict]:
    """Index every publicly included episode. ``episodes`` comes from ``discover_episodes``."""
    jobs = [
        e for e in episodes if _include_in_public_stats({"dataset_repo": e[0]}, e[0])
    ]
    rows: list[dict] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load_one, jobs, chunksize=32):
            if isinstance(result, dict):
                rows.append(result)
            elif result:
                errors.append(result)
    for msg in errors:
        print(f"  skipped {msg}", file=sys.stderr)
    print(f"indexed {len(rows)} episode(s), skipped {len(errors)}", file=sys.stderr)
    return rows


def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
