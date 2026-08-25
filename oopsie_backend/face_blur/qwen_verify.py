"""Verify deface-positive video frames with a vision-language model.

This pipeline consumes a completed Hugging Face face-presence scan, selects its
positive videos, reruns the same detector through every frame, and asks a
long-lived Qwen2.5-VL worker for a binary judgment on every detector-positive
full frame. It stores metadata and raw responses but no source or rendered
media.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from oopsie_backend.face_blur.cli import resolve_device
from oopsie_backend.face_blur.detectors import create_detector
from oopsie_backend.face_blur.hf_scan import (
    HfVideo,
    _atomic_csv,
    _atomic_json,
    canonical_hash,
    materialize_video,
    stable_video_id,
    utc_now,
    video_shard,
)
from oopsie_backend.face_blur.models import (
    default_model_dir,
    model_metadata,
    provision_model,
)

SCHEMA_VERSION = 1
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_PROMPT = """Decide whether this image frame visibly contains a human face.

Answer YES for a real human face, including a partial or profile face, or a
human face visible in a photograph, poster, reflection, or screen. Answer NO
for robot parts, grippers, hands or arms without a visible face, text, food,
containers, tools, shadows, and accidental face-like patterns. Do not infer a
face merely because a human body part is present. If a region is blurred and
no facial features are discernible, answer NO.

Respond with exactly one word: YES or NO."""


def parse_binary_response(raw_response: str) -> bool | None:
    """Parse a strict YES/NO response without treating ambiguity as negative."""
    tokens = re.findall(r"[a-z]+", raw_response.lower())
    answers = {token for token in tokens if token in {"yes", "no"}}
    if answers == {"yes"}:
        return True
    if answers == {"no"}:
        return False
    return None


def load_detector_candidates(
    detector_results_dir: Path,
    *,
    expected_detector: str,
    expected_threshold: float,
) -> tuple[dict, list[HfVideo]]:
    """Load successful positive videos from a completed detector scan."""
    detector_results_dir = detector_results_dir.resolve()
    config_path = detector_results_dir / "config.json"
    videos_path = detector_results_dir / "videos.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("detector") != expected_detector:
        raise ValueError(
            f"expected detector {expected_detector!r}, got {config.get('detector')!r}"
        )
    actual_threshold = float(config.get("score_threshold", -1))
    if abs(actual_threshold - expected_threshold) > 1e-12:
        raise ValueError(
            f"expected detector threshold {expected_threshold}, got {actual_threshold}"
        )
    with videos_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    candidates = []
    seen_video_ids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("status") != "ok" or row.get("has_face") != "True":
            continue
        video = HfVideo(
            repo_id=str(row["source_repo"]),
            repo_revision=str(row["repo_revision"]),
            source_path=str(row["source_path"]),
            episode_key=str(row["episode_key"]),
            video_id=str(row["video_id"]),
        )
        expected_id = stable_video_id(video.repo_id, video.source_path)
        if video.video_id != expected_id:
            raise ValueError(
                f"invalid video id for {video.repo_id}:{video.source_path}: "
                f"{video.video_id!r} != {expected_id!r}"
            )
        source = (video.repo_id, video.source_path)
        if video.video_id in seen_video_ids or source in seen_sources:
            raise ValueError(
                f"duplicate detector candidate: {video.repo_id}:{video.source_path}"
            )
        seen_video_ids.add(video.video_id)
        seen_sources.add(source)
        candidates.append(video)
    if not candidates:
        raise ValueError(f"detector scan has no positive videos: {videos_path}")
    candidates.sort(key=lambda video: (video.repo_id, video.source_path))
    return config, candidates


def detector_runtime_settings(config: dict) -> tuple[tuple[int, int], float]:
    """Extract the detector settings that must match the source scan."""
    raw_input_size = config.get("input_size")
    if not isinstance(raw_input_size, list) or len(raw_input_size) != 2:
        raise ValueError(f"invalid detector input_size: {raw_input_size!r}")
    input_size = tuple(int(value) for value in raw_input_size)
    if any(value <= 0 for value in input_size):
        raise ValueError(f"invalid detector input_size: {raw_input_size!r}")
    nms_threshold = float(config.get("nms_threshold", -1))
    if not 0 <= nms_threshold <= 1:
        raise ValueError(f"invalid detector nms_threshold: {nms_threshold}")
    return input_size, nms_threshold


class QwenWorker:
    """Persistent JSON-lines client for :mod:`qwen_worker`."""

    def __init__(
        self,
        *,
        worker_python: Path,
        model: str,
        revision: str,
        cache_dir: Path | None,
        device: str,
        min_visual_tokens: int,
        max_visual_tokens: int,
        max_new_tokens: int,
        local_files_only: bool,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        command = [
            # Preserve the virtual-environment entry point. ``resolve()`` follows
            # its python symlink to the base interpreter, which drops the venv's
            # site-packages in the spawned worker.
            str(worker_python.absolute()),
            "-m",
            "oopsie_backend.face_blur.qwen_worker",
            "--model",
            model,
            "--revision",
            revision,
            "--device",
            device,
            "--min-visual-tokens",
            str(min_visual_tokens),
            "--max-visual-tokens",
            str(max_visual_tokens),
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        if cache_dir is not None:
            command.extend(["--cache-dir", str(cache_dir.resolve())])
        if local_files_only:
            command.append("--local-files-only")
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(repo_root)
        )
        self.process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("could not create Qwen worker pipes")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        ready = self._read_message()
        if ready.get("type") != "ready" or ready.get("status") != "ok":
            error = ready.get("error") or "worker did not send a valid ready message"
            self.close(force=True)
            raise RuntimeError(f"Qwen worker initialization failed: {error}")
        self.metadata = dict(ready.get("metadata") or {})
        self.request_count = 0

    def _read_message(self) -> dict:
        line = self.stdout.readline()
        if not line:
            return_code = self.process.poll()
            raise RuntimeError(
                f"Qwen worker closed stdout unexpectedly (exit={return_code})"
            )
        message = json.loads(line)
        if not isinstance(message, dict):
            raise TypeError("Qwen worker returned a non-object message")
        return message

    def classify(self, encoded_images: list[str], *, prompt: str) -> list[str]:
        self.request_count += 1
        request_id = self.request_count
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "images": encoded_images,
        }
        self.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.stdin.flush()
        response = self._read_message()
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"Qwen worker response id mismatch: {response.get('request_id')} "
                f"!= {request_id}"
            )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Qwen worker inference failed: {response.get('error', 'unknown error')}"
            )
        raw_responses = list(response.get("raw_responses") or [])
        if len(raw_responses) != len(encoded_images):
            raise RuntimeError(
                f"Qwen worker returned {len(raw_responses)} responses for "
                f"{len(encoded_images)} images"
            )
        return [str(response) for response in raw_responses]

    def close(self, *, force: bool = False) -> None:
        if self.process.poll() is not None:
            return
        if not force:
            try:
                self.stdin.write(
                    json.dumps(
                        {"request_id": self.request_count + 1, "command": "shutdown"}
                    )
                    + "\n"
                )
                self.stdin.flush()
                self.process.wait(timeout=10)
                return
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def _atomic_json_gzip(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_json_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def can_resume(result: dict, video: HfVideo, config_hash: str) -> bool:
    source = result.get("source", {})
    return (
        result.get("status") == "ok"
        and result.get("config_hash") == config_hash
        and source.get("video_id") == video.video_id
        and source.get("repo_id") == video.repo_id
        and source.get("repo_revision") == video.repo_revision
        and source.get("source_path") == video.source_path
    )


def _encode_jpeg(frame: np.ndarray, quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("could not encode detector-positive frame as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def verify_video(
    path: Path,
    detector,
    worker: QwenWorker,
    *,
    prompt: str,
    jpeg_quality: int,
    batch_size: int,
    max_frames: int | None,
) -> dict:
    """Run deface exhaustively and verify every positive full frame with Qwen."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    detector_seconds = 0.0
    qwen_seconds = 0.0
    verification_frames: list[dict] = []
    pending: list[tuple[dict, str]] = []
    started = time.perf_counter()

    def flush_pending() -> None:
        nonlocal qwen_seconds
        if not pending:
            return
        inference_started = time.perf_counter()
        raw_responses = worker.classify(
            [encoded for _, encoded in pending],
            prompt=prompt,
        )
        qwen_seconds += time.perf_counter() - inference_started
        for (frame_result, _), raw_response in zip(pending, raw_responses):
            parsed = parse_binary_response(raw_response)
            frame_result["qwen_raw_response"] = raw_response
            frame_result["qwen_human_face"] = parsed
            verification_frames.append(frame_result)
        pending.clear()

    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            detector_started = time.perf_counter()
            detections = detector.detect(frame)
            detector_seconds += time.perf_counter() - detector_started
            if detections:
                frame_result = {
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp_ms, 3),
                    "deface_detections": [
                        detection.to_dict() for detection in detections
                    ],
                    "deface_max_score": max(
                        float(detection.score) for detection in detections
                    ),
                }
                pending.append((frame_result, _encode_jpeg(frame, jpeg_quality)))
                if len(pending) >= batch_size:
                    flush_pending()
            frame_index += 1
        flush_pending()
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"video decoded zero frames: {path}")
    qwen_positive = sum(
        frame.get("qwen_human_face") is True for frame in verification_frames
    )
    qwen_negative = sum(
        frame.get("qwen_human_face") is False for frame in verification_frames
    )
    qwen_unparsed = len(verification_frames) - qwen_positive - qwen_negative
    # A decode that ends before the container's advertised frame count is not
    # safe to treat as a Qwen rejection. Debug-limited runs are likewise
    # incomplete unless the limit happens to cover the full video.
    if expected_frames > 0:
        partial = frame_index < expected_frames
    else:
        partial = max_frames is not None and frame_index >= max_frames
    if qwen_positive:
        verification_status = "verified"
        verified_face: bool | None = True
    elif not verification_frames or qwen_unparsed or partial:
        verification_status = "incomplete"
        verified_face = None
    else:
        verification_status = "rejected"
        verified_face = False
    elapsed_seconds = time.perf_counter() - started
    return {
        "verification_status": verification_status,
        "qwen_verified_face": verified_face,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "expected_frames": expected_frames,
            "decoded_frames": frame_index,
            "partial": partial,
        },
        "counts": {
            "deface_positive_frames": len(verification_frames),
            "qwen_positive_frames": qwen_positive,
            "qwen_negative_frames": qwen_negative,
            "qwen_unparsed_frames": qwen_unparsed,
            "deface_detections": sum(
                len(frame["deface_detections"]) for frame in verification_frames
            ),
        },
        "performance": {
            "elapsed_seconds": elapsed_seconds,
            "detector_seconds": detector_seconds,
            "qwen_seconds": qwen_seconds,
            "decoded_fps": frame_index / elapsed_seconds if elapsed_seconds else None,
            "qwen_frames_per_second": (
                len(verification_frames) / qwen_seconds if qwen_seconds else None
            ),
        },
        "verification_frames": verification_frames,
    }


def result_to_row(result: dict) -> dict:
    source = result.get("source", {})
    verification = result.get("verification", {})
    counts = verification.get("counts", {})
    video = verification.get("video", {})
    performance = verification.get("performance", {})
    status = result.get("status")
    return {
        "video_id": source.get("video_id"),
        "source_repo": source.get("repo_id"),
        "repo_revision": source.get("repo_revision"),
        "source_path": source.get("source_path"),
        "episode_key": source.get("episode_key"),
        "status": status,
        "error": result.get("error"),
        "verification_status": (
            verification.get("verification_status") if status == "ok" else None
        ),
        "qwen_verified_face": (
            verification.get("qwen_verified_face") if status == "ok" else None
        ),
        "deface_positive_frames": counts.get("deface_positive_frames"),
        "qwen_positive_frames": counts.get("qwen_positive_frames"),
        "qwen_negative_frames": counts.get("qwen_negative_frames"),
        "qwen_unparsed_frames": counts.get("qwen_unparsed_frames"),
        "deface_detections": counts.get("deface_detections"),
        "decoded_frames": video.get("decoded_frames"),
        "expected_frames": video.get("expected_frames"),
        "partial": video.get("partial"),
        "elapsed_seconds": performance.get("elapsed_seconds"),
        "detector_seconds": performance.get("detector_seconds"),
        "qwen_seconds": performance.get("qwen_seconds"),
        "qwen_frames_per_second": performance.get("qwen_frames_per_second"),
        "completed_at": result.get("completed_at"),
    }


def aggregate_episode_rows(video_rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in video_rows:
        groups[(str(row.get("source_repo")), str(row.get("episode_key")))].append(row)
    episode_rows = []
    for (repo_id, episode_key), rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("status") == "ok"]
        verified = [row for row in successful if row.get("qwen_verified_face") is True]
        incomplete = [
            row for row in successful if row.get("qwen_verified_face") is None
        ]
        failed = len(rows) - len(successful)
        if verified:
            verification_status = "verified"
            qwen_verified_face: bool | None = True
        elif incomplete or failed:
            verification_status = "incomplete"
            qwen_verified_face = None
        else:
            verification_status = "rejected"
            qwen_verified_face = False
        episode_rows.append(
            {
                "source_repo": repo_id,
                "episode_key": episode_key,
                "verification_status": verification_status,
                "qwen_verified_face": qwen_verified_face,
                "video_count": len(rows),
                "verified_video_count": len(verified),
                "rejected_video_count": sum(
                    row.get("qwen_verified_face") is False for row in successful
                ),
                "incomplete_video_count": len(incomplete) + failed,
                "deface_positive_frame_count": sum(
                    int(row.get("deface_positive_frames") or 0) for row in successful
                ),
                "qwen_positive_frame_count": sum(
                    int(row.get("qwen_positive_frames") or 0) for row in successful
                ),
                "source_paths_json": json.dumps(
                    sorted(str(row.get("source_path")) for row in rows),
                    separators=(",", ":"),
                ),
                "verified_source_paths_json": json.dumps(
                    sorted(str(row.get("source_path")) for row in verified),
                    separators=(",", ":"),
                ),
            }
        )
    return episode_rows


def write_reports(output_dir: Path, results: list[dict], config: dict) -> dict:
    video_rows = sorted(
        (result_to_row(result) for result in results),
        key=lambda row: (str(row.get("source_repo")), str(row.get("source_path"))),
    )
    episode_rows = aggregate_episode_rows(video_rows)
    _atomic_csv(output_dir / "videos.csv", video_rows)
    _atomic_csv(output_dir / "episodes.csv", episode_rows)
    successful = [row for row in video_rows if row.get("status") == "ok"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": config,
        "totals": {
            "candidate_videos": len(video_rows),
            "successful_videos": len(successful),
            "failed_videos": len(video_rows) - len(successful),
            "verified_videos": sum(
                row.get("qwen_verified_face") is True for row in successful
            ),
            "rejected_videos": sum(
                row.get("qwen_verified_face") is False for row in successful
            ),
            "incomplete_videos": sum(
                row.get("qwen_verified_face") is None for row in successful
            ),
            "candidate_episodes": len(episode_rows),
            "verified_episodes": sum(
                row.get("qwen_verified_face") is True for row in episode_rows
            ),
            "deface_positive_frames": sum(
                int(row.get("deface_positive_frames") or 0) for row in successful
            ),
            "qwen_positive_frames": sum(
                int(row.get("qwen_positive_frames") or 0) for row in successful
            ),
            "qwen_unparsed_frames": sum(
                int(row.get("qwen_unparsed_frames") or 0) for row in successful
            ),
        },
        "outputs": {
            "videos": "videos.csv",
            "episodes": "episodes.csv",
            "results": "results/*.json.gz",
        },
    }
    _atomic_json(output_dir / "scan-report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--detector-threshold", type=float, default=0.4)
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-revision", default=DEFAULT_QWEN_REVISION)
    parser.add_argument("--qwen-cache-dir", type=Path, default=None)
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug only.")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.detector_threshold <= 1:
        parser.error("--detector-threshold must be between 0 and 1")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard count or index")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    try:
        detector_source_config, candidates = load_detector_candidates(
            args.detector_results_dir,
            expected_detector="deface-centerface",
            expected_threshold=args.detector_threshold,
        )
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        print(f"error: detector candidates: {type(exc).__name__}: {exc}")
        return 1
    all_candidate_count = len(candidates)
    candidates = [
        video
        for video in candidates
        if video_shard(video, args.num_shards) == args.shard_index
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("error: no detector-positive videos in this shard")
        return 1

    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        detector_input_size, detector_nms_threshold = detector_runtime_settings(
            detector_source_config
        )
        detector_device = resolve_device(args.detector_device)
        detector_model_path = provision_model("deface-centerface", args.model_dir)
        detector_model_info = model_metadata("deface-centerface", detector_model_path)
        source_model_sha = detector_source_config.get("model_sha256")
        if source_model_sha and source_model_sha != detector_model_info["model_sha256"]:
            raise ValueError(
                "deface model differs from source scan: "
                f"{detector_model_info['model_sha256']} != {source_model_sha}"
            )
        detector = create_detector(
            "deface-centerface",
            detector_model_path,
            input_size=detector_input_size,
            score_threshold=args.detector_threshold,
            nms_threshold=detector_nms_threshold,
            device=detector_device,
        )
        detector.detect(
            np.zeros(
                (detector_input_size[1], detector_input_size[0], 3),
                dtype=np.uint8,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        print(f"error: detector initialization: {type(exc).__name__}: {exc}")
        return 1

    worker: QwenWorker | None = None
    try:
        worker = QwenWorker(
            worker_python=args.worker_python,
            model=args.qwen_model,
            revision=args.qwen_revision,
            cache_dir=args.qwen_cache_dir,
            device=args.qwen_device,
            min_visual_tokens=args.min_visual_tokens,
            max_visual_tokens=args.max_visual_tokens,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.qwen_local_files_only,
        )
        raw_config = {
            "schema_version": SCHEMA_VERSION,
            "detector_source_dir": str(args.detector_results_dir.resolve()),
            "detector_source_config_hash": detector_source_config.get("config_hash"),
            "detector": "deface-centerface",
            "detector_model_sha256": detector_model_info["model_sha256"],
            "detector_threshold": args.detector_threshold,
            "detector_input_size": list(detector_input_size),
            "detector_nms_threshold": detector_nms_threshold,
            "detector_device": detector_device,
            "qwen_model": args.qwen_model,
            "qwen_requested_revision": args.qwen_revision,
            "qwen_resolved_revision": worker.metadata.get("resolved_revision"),
            "qwen_device": args.qwen_device,
            "qwen_worker_metadata": worker.metadata,
            "prompt": args.prompt,
            "min_visual_tokens": args.min_visual_tokens,
            "max_visual_tokens": args.max_visual_tokens,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "jpeg_quality": args.jpeg_quality,
            "max_frames": args.max_frames,
        }
        config_hash = canonical_hash(raw_config)
        config = {**raw_config, "config_hash": config_hash}
        _atomic_json(output_dir / "config.json", config)

        results = []
        for video in tqdm(candidates, desc="Qwen face verification", unit="video"):
            result_path = result_dir / f"{video.video_id}.json.gz"
            if not args.no_resume and result_path.is_file():
                try:
                    cached = _load_json_gzip(result_path)
                except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile):
                    cached = {}
                if can_resume(cached, video, config_hash):
                    results.append(cached)
                    continue
            source = {
                "video_id": video.video_id,
                "repo_id": video.repo_id,
                "repo_revision": video.repo_revision,
                "source_path": video.source_path,
                "episode_key": video.episode_key,
            }
            try:
                with materialize_video(
                    video,
                    token=args.token,
                    download_mode="temporary",
                    cache_dir=None,
                    temp_dir=args.temp_dir,
                    local_root=None,
                ) as (local_path, materialization):
                    verification = verify_video(
                        local_path,
                        detector,
                        worker,
                        prompt=args.prompt,
                        jpeg_quality=args.jpeg_quality,
                        batch_size=args.batch_size,
                        max_frames=args.max_frames,
                    )
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "config_hash": config_hash,
                    "config": config,
                    "source": source,
                    "materialization": materialization,
                    "verification": verification,
                    "completed_at": utc_now(),
                }
            except Exception as exc:  # noqa: BLE001 - preserve per-video progress
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "config_hash": config_hash,
                    "config": config,
                    "source": source,
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at": utc_now(),
                }
                tqdm.write(f"{video.repo_id}:{video.source_path}: {result['error']}")
            _atomic_json_gzip(result_path, result)
            results.append(result)
    except Exception as exc:  # noqa: BLE001 - worker/process error boundary
        print(f"error: Qwen verifier: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if worker is not None:
            worker.close()

    report = write_reports(output_dir, results, config)
    report["source"] = {
        "all_detector_candidate_videos": all_candidate_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    _atomic_json(output_dir / "scan-report.json", report)
    totals = report["totals"]
    print(
        f"verified {totals['verified_videos']} / "
        f"{totals['candidate_videos']} candidate videos; "
        f"rejected {totals['rejected_videos']}; "
        f"incomplete {totals['incomplete_videos']}; "
        f"failed {totals['failed_videos']}"
    )
    return 1 if totals["failed_videos"] or totals["incomplete_videos"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
