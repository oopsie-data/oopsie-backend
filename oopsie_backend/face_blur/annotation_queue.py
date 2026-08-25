"""Prepare a resumable manual face/privacy annotation workspace."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from oopsie_backend.face_blur.hf_scan import stable_video_id, utc_now

FALSE_POSITIVE_REVIEW_IDS = {
    "video_7788ff8f79c94e89",
    "video_4f88c2595bca2616",
}
FALSE_NEGATIVE_REVIEW_IDS = {
    "video_8af18fec6354aba7",
    "video_710d65325278ff7c",
    "video_7403d0eecd45c241",
}
TASK_RELEVANT_TEST_IDS = {"video_0172"}

FIELDNAMES = [
    "sample_id",
    "source_kind",
    "source_repo",
    "repo_revision",
    "source_path",
    "episode_key",
    "local_video_path",
    "face_present",
    "blur_action",
    "exemption_reason",
    "annotation_source",
    "annotated_at",
    "labeler",
    "notes",
    "candidate_reason",
    "qwen_verified_face",
    "deface_positive_frames",
    "qwen_positive_frames",
    "qwen_negative_frames",
    "deface_detections",
    "disagreement_rank_repo",
    "video_storage",
]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "video.mp4"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _append_reason(row: dict[str, str], reason: str) -> None:
    reasons = set(filter(None, row.get("candidate_reason", "").split("|")))
    reasons.add(reason)
    row["candidate_reason"] = "|".join(sorted(reasons))


def _blank_row(repo_id: str, source_path: str) -> dict[str, str]:
    video_id = stable_video_id(repo_id, source_path)
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "sample_id": video_id,
            "source_kind": "huggingface",
            "source_repo": repo_id,
            "source_path": source_path,
        }
    )
    return row


def _apply_test_label(target: dict[str, str], source: dict[str, str]) -> None:
    label = (source.get("face_present") or "").strip().lower()
    notes = (source.get("notes") or "").strip()
    target["annotation_source"] = "imported_testset"
    target["labeler"] = (source.get("labeler") or "").strip()
    target["notes"] = notes
    if label == "no":
        target.update(face_present="no", blur_action="preserve", exemption_reason="")
    elif label == "yes":
        target.update(face_present="yes", blur_action="", exemption_reason="")
    elif label == "uncertain" and "already blurred" in notes.lower():
        target.update(
            face_present="yes",
            blur_action="preserve",
            exemption_reason="already_blurred",
        )
    elif label == "uncertain":
        target.update(face_present="uncertain", blur_action="review", exemption_reason="")
    if source.get("sample_id") in TASK_RELEVANT_TEST_IDS:
        target.update(
            face_present="yes",
            blur_action="preserve",
            exemption_reason="task_relevant_depiction",
        )


def _apply_recent_review_label(target: dict[str, str], source: dict[str, str]) -> None:
    video_id = source["video_id"]
    predicted_face = source["verdict"] == "accepted"
    if video_id in FALSE_POSITIVE_REVIEW_IDS:
        face_present = False
    elif video_id in FALSE_NEGATIVE_REVIEW_IDS:
        face_present = True
    else:
        face_present = predicted_face
    previous_special_action = (
        target.get("blur_action") == "preserve"
        and target.get("exemption_reason")
        in {"already_blurred", "task_relevant_depiction"}
    )
    target["face_present"] = "yes" if face_present else "no"
    if not face_present:
        target["blur_action"] = "preserve"
        target["exemption_reason"] = ""
    elif not previous_special_action:
        target["blur_action"] = ""
        target["exemption_reason"] = ""
    target["annotation_source"] = "imported_manual_review_2026-08-11"
    target["annotated_at"] = "2026-08-11"
    statement = (
        "Manual Qwen review set: face present."
        if face_present
        else "Manual Qwen review set: no face."
    )
    existing = target.get("notes", "").strip()
    target["notes"] = f"{existing} {statement}".strip()


def build_queue(
    *,
    qwen_videos: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
    disagreement_per_repo: int,
) -> tuple[list[dict[str, str]], dict[tuple[str, str], Path]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    local_sources: dict[tuple[str, str], Path] = {}
    inventory_revision = {
        (row.get("source_repo") or row.get("repo_id") or "", row["source_path"]): row.get("repo_revision", "")
        for row in inventory_rows
    }

    def row_for(repo_id: str, source_path: str) -> dict[str, str]:
        key = (repo_id, source_path)
        if key not in by_key:
            by_key[key] = _blank_row(*key)
            by_key[key]["repo_revision"] = inventory_revision.get(key, "")
        return by_key[key]

    rejected_by_repo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in qwen_videos:
        if source["qwen_verified_face"] == "False":
            rejected_by_repo[source["source_repo"]].append(source)
    disagreement_keys: dict[tuple[str, str], int] = {}
    for repo_id, candidates in rejected_by_repo.items():
        ranked = sorted(
            candidates,
            key=lambda row: (
                int(row["deface_positive_frames"]),
                int(row["deface_detections"]),
                row["video_id"],
            ),
            reverse=True,
        )
        for rank, source in enumerate(ranked[:disagreement_per_repo], start=1):
            disagreement_keys[(repo_id, source["source_path"])] = rank

    qwen_by_key = {
        (row["source_repo"], row["source_path"]): row for row in qwen_videos
    }
    for key, source in qwen_by_key.items():
        accepted = source["qwen_verified_face"] == "True"
        disagreement_rank = disagreement_keys.get(key)
        if not accepted and disagreement_rank is None:
            continue
        target = row_for(*key)
        target.update(
            {
                "sample_id": source["video_id"],
                "repo_revision": source["repo_revision"],
                "episode_key": source["episode_key"],
                "qwen_verified_face": source["qwen_verified_face"],
                "deface_positive_frames": source["deface_positive_frames"],
                "qwen_positive_frames": source["qwen_positive_frames"],
                "qwen_negative_frames": source["qwen_negative_frames"],
                "deface_detections": source["deface_detections"],
                "disagreement_rank_repo": str(disagreement_rank or ""),
            }
        )
        _append_reason(
            target,
            "qwen_accepted" if accepted else "high_deface_qwen_disagreement",
        )

    for source in test_rows:
        key = (source["source_repo"], source["source_path"])
        target = row_for(*key)
        _append_reason(target, "existing_testset")
        _apply_test_label(target, source)
        local_sources[key] = (
            Path(source["_manifest_root"]) / source["local_video_path"]
        ).resolve()
        if key in qwen_by_key:
            model = qwen_by_key[key]
            target.update(
                {
                    "sample_id": model["video_id"],
                    "repo_revision": model["repo_revision"],
                    "episode_key": model["episode_key"],
                    "qwen_verified_face": model["qwen_verified_face"],
                    "deface_positive_frames": model["deface_positive_frames"],
                    "qwen_positive_frames": model["qwen_positive_frames"],
                    "qwen_negative_frames": model["qwen_negative_frames"],
                    "deface_detections": model["deface_detections"],
                }
            )

    for source in review_rows:
        key = (source["source_repo"], source["source_path"])
        target = row_for(*key)
        _append_reason(target, "manual_review_2026-08-05")
        _apply_recent_review_label(target, source)
        local_sources[key] = (
            Path(source["_manifest_root"]) / source["local_video_path"]
        ).resolve()
        if key in qwen_by_key:
            model = qwen_by_key[key]
            target.update(
                {
                    "sample_id": model["video_id"],
                    "repo_revision": model["repo_revision"],
                    "episode_key": model["episode_key"],
                    "qwen_verified_face": model["qwen_verified_face"],
                    "deface_positive_frames": model["deface_positive_frames"],
                    "qwen_positive_frames": model["qwen_positive_frames"],
                    "qwen_negative_frames": model["qwen_negative_frames"],
                    "deface_detections": model["deface_detections"],
                }
            )

    rows = list(by_key.values())
    reason_priority = {
        "qwen_accepted": 0,
        "high_deface_qwen_disagreement": 1,
        "manual_review_2026-08-05": 2,
        "existing_testset": 3,
    }
    rows.sort(
        key=lambda row: (
            min(
                reason_priority.get(reason, 99)
                for reason in row["candidate_reason"].split("|")
            ),
            row["source_repo"],
            -int(row["deface_positive_frames"] or 0),
            row["source_path"],
        )
    )
    return rows, local_sources


def materialize_videos(
    rows: list[dict[str, str]],
    local_sources: dict[tuple[str, str], Path],
    *,
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
) -> int:
    failures = 0
    for index, row in enumerate(rows, start=1):
        repo_name = row["source_repo"].rsplit("/", 1)[-1]
        relative = Path("videos") / repo_name / (
            f"{row['sample_id']}__{_safe_name(Path(row['source_path']).name)}"
        )
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        row["local_video_path"] = relative.as_posix()
        if destination.is_file():
            row["video_storage"] = "existing"
            continue
        key = (row["source_repo"], row["source_path"])
        try:
            source = local_sources.get(key)
            if source is None:
                source = Path(
                    hf_hub_download(
                        repo_id=row["source_repo"],
                        filename=row["source_path"],
                        repo_type="dataset",
                        revision=row["repo_revision"] or None,
                        cache_dir=cache_dir,
                        token=token,
                    )
                ).resolve()
                source_kind = "hf_cache"
            else:
                if not source.is_file():
                    raise FileNotFoundError(f"imported local video missing: {source}")
                source_kind = "imported_local"
            try:
                os.link(source.resolve(), destination)
                row["video_storage"] = f"hardlink_{source_kind}"
            except OSError:
                shutil.copy2(source, destination)
                row["video_storage"] = f"copy_{source_kind}"
            print(
                f"[{index}/{len(rows)}] {row['source_repo']}:{row['source_path']}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            row["video_storage"] = f"error:{type(exc).__name__}:{exc}"
            print(
                f"[{index}/{len(rows)}] ERROR {row['source_repo']}:"
                f"{row['source_path']}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if index % 10 == 0:
            _atomic_csv(output_dir / "annotations.csv", rows)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-videos", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--disagreement-per-repo", type=int, default=20)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.disagreement_per_repo <= 0:
        raise SystemExit("--disagreement-per-repo must be positive")
    test_rows = _read_csv(args.test_manifest)
    for row in test_rows:
        row["_manifest_root"] = str(args.test_manifest.resolve().parent)
    review_rows = _read_csv(args.review_manifest)
    for row in review_rows:
        row["_manifest_root"] = str(args.review_manifest.resolve().parent)
    rows, local_sources = build_queue(
        qwen_videos=_read_csv(args.qwen_videos),
        test_rows=test_rows,
        review_rows=review_rows,
        inventory_rows=_read_csv(args.inventory),
        disagreement_per_repo=args.disagreement_per_repo,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "annotations.csv", rows)
    failures = materialize_videos(
        rows,
        local_sources,
        output_dir=output_dir,
        cache_dir=args.cache_dir.resolve(),
        token=args.token,
    )
    _atomic_csv(output_dir / "annotations.csv", rows)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "queue": {
            "videos": len(rows),
            "qwen_accepted": sum("qwen_accepted" in r["candidate_reason"] for r in rows),
            "high_deface_qwen_disagreement": sum(
                "high_deface_qwen_disagreement" in r["candidate_reason"] for r in rows
            ),
            "existing_testset": sum("existing_testset" in r["candidate_reason"] for r in rows),
            "recent_manual_review": sum(
                "manual_review_2026-08-05" in r["candidate_reason"] for r in rows
            ),
        },
        "annotations": {
            "face_yes": sum(r["face_present"] == "yes" for r in rows),
            "face_no": sum(r["face_present"] == "no" for r in rows),
            "uncertain": sum(r["face_present"] == "uncertain" for r in rows),
            "unlabeled": sum(not r["face_present"] for r in rows),
            "blur": sum(r["blur_action"] == "blur" for r in rows),
            "preserve": sum(r["blur_action"] == "preserve" for r in rows),
            "review": sum(r["blur_action"] == "review" for r in rows),
            "needs_action": sum(
                r["face_present"] == "yes" and not r["blur_action"] for r in rows
            ),
        },
        "download_failures": failures,
        "disagreement_per_repo": args.disagreement_per_repo,
        "manifest": "annotations.csv",
        "history": "annotations.history.jsonl",
    }
    _atomic_json(output_dir / "setup-report.json", report)
    (output_dir / "README.md").write_text(
        "# Manual face/privacy annotation workspace\n\n"
        "`annotations.csv` is the authoritative manifest. It records face presence "
        "separately from the downstream `blur_action`. The browser UI appends every "
        "manual edit to `annotations.history.jsonl`.\n\n"
        "Decision shortcuts: `b` face/blur, `n` no face, `a` already blurred, "
        "`p` task-relevant photo/depiction, `o` other preserve, `u` uncertain/review.\n\n"
        "After annotation, the same manifest can be passed to any face-blur runner "
        "with `--blur-actions blur`; preserved and review rows will be excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
