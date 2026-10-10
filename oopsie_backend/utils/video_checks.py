"""Video quality checks, run as the ``video`` category of the action QA
(``oopsie_backend/scripts/run_action_qa.py``).

Each camera video is decoded once with ffmpeg into small grayscale frames; all
checks run on that probe:

- ``resolution``: short side below ``min_short_side``.
- ``decoding``: missing file, ffmpeg errors, nothing decoded, or the decoded frame
  count differing from the container's.
- ``brightness``: median luma too dark or too bright, or a large share of
  near-black / near-white frames.
- ``speed``: container fps differs from ``control_freq``, video duration differs
  from the trajectory duration, irregular frame timestamps, or the cameras of one
  episode disagreeing in duration.
- ``heavy_artifacts_and_desync``: frozen runs or many duplicate frames while the
  robot moves, image jumps (sudden one-way changes), and single corrupted frames.
- ``periodic_skips``: recording pauses at a regular period (e.g. during each
  action-chunk inference) while the low-level controller keeps the robot moving,
  seen as robot state that jumps at that period throughout the motion.
  Episode-level, from the robot state.

Frame differences are mean absolute differences of 0-255 grayscale values.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from oopsie_backend.utils.action_qa import _finding, _params, _skip

CATEGORY = "video"
ARTIFACTS = "heavy_artifacts_and_desync"
CHECKS = ("resolution", "decoding", "brightness", "speed", ARTIFACTS, "periodic_skips")
ALL_CAMERAS = "all cameras"
PROBE_SIZE = 96
_PTS_RE = re.compile(r"pts_time:\s*([-+\d.eE]+)")
_ERROR_RE = re.compile(r"\[(error|fatal)\]")
SKIP_WINDOW = 8
SKIP_MAX_PERIOD = 30


@dataclass
class VideoProbe:
    width: int
    height: int
    fps: float
    container_frames: int
    returncode: int
    errors: list[str] = field(default_factory=list)
    pts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    luma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diff: np.ndarray = field(default_factory=lambda: np.zeros(0))
    skip_diff: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def decoded_frames(self) -> int:
        return len(self.luma)

    @property
    def duration(self) -> float | None:
        return self.decoded_frames / self.fps if self.fps > 0 and self.decoded_frames else None


def probe_video(path: Path, size: int = PROBE_SIZE) -> VideoProbe:
    """Read container metadata and decode the full stream to ``size``x``size`` gray."""
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "level+info",
        "-i",
        str(path),
        "-fps_mode",
        "passthrough",
        "-vf",
        f"scale={size}:{size},format=gray,showinfo",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    log = proc.stderr.decode("utf-8", errors="replace").splitlines()
    errors = [line.strip() for line in log if _ERROR_RE.search(line)]
    pts = np.array([float(m.group(1)) for line in log if (m := _PTS_RE.search(line))])

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = raw.size // (size * size)
    frames = raw[: n * size * size].reshape(n, size, size).astype(np.float32)
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        container_frames=container_frames,
        returncode=proc.returncode,
        errors=errors,
        pts=pts,
        luma=frames.mean(axis=(1, 2)),
        diff=np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)),
        skip_diff=np.abs(frames[2:] - frames[:-2]).mean(axis=(1, 2)),
    )


def state_step(states: dict[str, np.ndarray]) -> np.ndarray | None:
    """Per-step size of the robot state change, over all non-gripper states, each
    scaled by its standard deviation. None without usable state."""
    arrays = [
        a.reshape(len(a), -1)
        for k, a in states.items()
        if k != "gripper_position" and a.ndim >= 1 and len(a) >= 2
    ]
    if not arrays:
        return None
    n = min(len(a) for a in arrays)
    state = np.concatenate([a[:n] for a in arrays], axis=1)
    scale = state.std(axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return np.nan_to_num(np.linalg.norm(np.diff(state / scale, axis=0), axis=1))


def robot_moving(step: np.ndarray, moving_fraction_of_p95: float) -> np.ndarray:
    """Mask of steps above ``moving_fraction_of_p95`` of the 95th percentile step."""
    p95 = np.percentile(step, 95)
    if p95 <= 0:
        return np.zeros(len(step), dtype=bool)
    return step > moving_fraction_of_p95 * p95


def _resample(mask: np.ndarray, n: int) -> np.ndarray:
    if len(mask) == n or len(mask) == 0:
        return mask[:n]
    idx = np.round(np.linspace(0, len(mask) - 1, n)).astype(int)
    return mask[idx]


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    padded = np.pad(x, w, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1), axis=1)


def _best_period(gaps: np.ndarray) -> tuple[int, float]:
    """Period ``p`` that best explains the gaps between events, as the share of gaps
    within one step of a multiple of ``p`` minus that share for random gaps (3/p).
    Among periods within 0.05 of the best score, the one with most exact multiples wins."""
    scored = []
    for p in range(3, SKIP_MAX_PERIOD + 1):
        r = gaps % p
        near = float((np.minimum(r, p - r) <= 1).mean()) - 3.0 / p
        scored.append((p, near, float((r == 0).mean())))
    top = max(s[1] for s in scored)
    if top <= 0:
        return 0, 0.0
    p, near, _ = max((s for s in scored if s[1] >= top - 0.05), key=lambda s: s[2])
    return p, near


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def check_resolution(p: VideoProbe, min_short_side: int) -> list[str]:
    if 0 < min(p.width, p.height) < min_short_side:
        return [f"{p.width}x{p.height} (short side < {min_short_side})"]
    return []


def check_decoding(p: VideoProbe, max_frame_count_diff: int) -> list[str]:
    issues = []
    if p.decoded_frames == 0:
        issues.append(f"no frames decoded (ffmpeg exit {p.returncode})")
    elif p.returncode != 0:
        issues.append(f"ffmpeg exit {p.returncode}")
    if p.errors:
        issues.append(f"{len(p.errors)} decoder error(s), first: {p.errors[0][:200]}")
    if (
        p.decoded_frames
        and p.container_frames > 0
        and abs(p.decoded_frames - p.container_frames) > max_frame_count_diff
    ):
        issues.append(f"decoded {p.decoded_frames} frames, container reports {p.container_frames}")
    return issues


def check_brightness(
    p: VideoProbe,
    dark_median: float,
    bright_median: float,
    dark_frame: float,
    bright_frame: float,
    extreme_frame_fraction: float,
) -> list[str]:
    if not p.decoded_frames:
        return []
    issues = []
    median = float(np.median(p.luma))
    if median < dark_median:
        issues.append(f"too dark: median luma {median:.0f}")
    elif median > bright_median:
        issues.append(f"too bright: median luma {median:.0f}")
    dark = float((p.luma < dark_frame).mean())
    bright = float((p.luma > bright_frame).mean())
    if dark >= extreme_frame_fraction:
        issues.append(f"{dark:.0%} of frames near black")
    if bright >= extreme_frame_fraction:
        issues.append(f"{bright:.0%} of frames near white")
    return issues


def check_speed(
    p: VideoProbe,
    n_steps: int | None,
    control_freq: float | None,
    fps_rel_tol: float,
    duration_tol_s: float,
    duration_rel_tol: float,
    frame_interval_cv: float,
) -> list[str]:
    issues = []
    if control_freq and p.fps > 0 and abs(p.fps - control_freq) > fps_rel_tol * control_freq:
        issues.append(f"video fps {p.fps:g} != control_freq {control_freq:g}")
    if control_freq and n_steps and p.duration is not None:
        expected = n_steps / control_freq
        if abs(p.duration - expected) > max(duration_tol_s, duration_rel_tol * expected):
            issues.append(
                f"video lasts {p.duration:.1f}s, trajectory {expected:.1f}s "
                f"({p.decoded_frames} frames, {n_steps} steps)"
            )
    if len(p.pts) > 2:
        dt = np.diff(p.pts)
        mean = dt.mean()
        if mean > 0 and dt.std() / mean > frame_interval_cv:
            issues.append(
                f"irregular frame timestamps: interval {mean * 1000:.0f}±{dt.std() * 1000:.0f} ms"
            )
    return issues


def check_frames(
    p: VideoProbe,
    moving: np.ndarray | None,
    rate: float,
    static_diff: float,
    duplicate_diff: float,
    frozen_s: float,
    choppy_fraction: float,
    jump_diff: float,
    jump_ratio: float,
    glitch_diff: float,
    glitch_ratio: float,
) -> list[str]:
    """Frozen/choppy need ``moving`` (robot-state motion); jumps and glitches do not."""
    d = p.diff
    if len(d) < 3:
        return []
    issues = []
    rate = rate if rate > 0 else 1.0

    if moving is not None and moving.any():
        mov = _resample(moving, len(d))
        frozen = _longest_run((d < static_diff) & mov)
        if frozen / rate >= frozen_s:
            issues.append(f"frozen for {frozen / rate:.1f}s while the robot moves")
        dup = float(((d < duplicate_diff) & mov).sum() / mov.sum())
        if dup >= choppy_fraction:
            issues.append(f"{dup:.0%} duplicate frames while the robot moves (choppy)")

    window = max(2, round(rate))
    glitch_at = set()
    for t in range(1, len(d)):
        a = min(d[t - 1], d[t])
        if a >= glitch_diff and a / (p.skip_diff[t - 1] + 1.0) >= glitch_ratio:
            glitch_at.update({t - 1, t})
            issues.append(f"corrupted frame at {t / rate:.1f}s")
    for t in range(len(d)):
        if t in glitch_at or d[t] < jump_diff:
            continue
        neighbours = np.r_[d[max(0, t - window) : t], d[t + 1 : t + 1 + window]]
        if neighbours.size and d[t] / (np.median(neighbours) + 1.0) >= jump_ratio:
            issues.append(f"image jump at {(t + 1) / rate:.1f}s")
    return issues


def check_periodic_skips(
    step: np.ndarray,
    moving: np.ndarray,
    control_freq: float | None,
    jump_ratio: float,
    min_jumps: int,
    min_period_score: float,
    min_coverage: float,
) -> list[str]:
    """Recording skips that drop time from state and video alike: isolated one-step
    jumps of the robot state at a regular period throughout the motion.

    A jump is a step above ``jump_ratio`` times the local median step (window of
    +-SKIP_WINDOW steps) while the robot moves, with no jump on either neighbour.
    Flags when at least ``min_jumps`` jumps have a period score of at least
    ``min_period_score`` and cover at least ``min_coverage`` of the periods while
    the robot moves (jumps x period / moving steps).
    """
    if len(step) < 2 * SKIP_WINDOW + 1 or not moving.any():
        return []
    med = _rolling_median(step, SKIP_WINDOW)
    jump = (step > jump_ratio * med) & moving
    jump &= ~(np.r_[False, jump[:-1]] | np.r_[jump[1:], False])
    idx = np.flatnonzero(jump)
    if len(idx) < min_jumps:
        return []
    period, score = _best_period(np.diff(idx))
    if score < min_period_score:
        return []
    coverage = len(idx) * period / moving.sum()
    if coverage < min_coverage:
        return []
    every = f"{period} steps" + (f" ({period / control_freq:.1f}s)" if control_freq else "")
    issue = (
        f"periodic skips every {every}: {len(idx)} jumps in robot state, "
        f"in {min(coverage, 1.0):.0%} of the periods while the robot moves"
    )
    return [issue]


def run_video_checks(
    videos: dict[str, Path],
    states: dict[str, np.ndarray],
    control_freq: float | None,
    cfg: dict,
) -> dict:
    """Run every enabled video check. ``videos`` maps camera name to mp4 path,
    ``cfg`` is the ``video`` section of the action QA config.

    Returns ``{"findings", "skips", "checked"}`` like ``run_action_qa_on_episode``;
    findings are keyed by camera and carry the readable ``issue``.
    """
    enabled = [test for test in CHECKS if cfg[test]["enabled"]]
    findings: list[dict] = []
    skips: list[dict] = []
    if not enabled:
        return {"findings": findings, "skips": skips, "checked": []}
    if not videos:
        skips += [_skip(CATEGORY, t, ALL_CAMERAS, "no_video_files") for t in enabled]
        return {"findings": findings, "skips": skips, "checked": []}

    def add(test: str, camera: str, issues: list[str]) -> None:
        findings.extend(_finding(CATEGORY, test, camera, issue=issue) for issue in issues)

    n_steps = next((len(a) for a in states.values() if a.ndim >= 1), None)
    moving = None
    artifacts_params = _params(cfg[ARTIFACTS])
    moving_fraction = artifacts_params.pop("moving_fraction_of_p95")
    state_tests = [t for t in (ARTIFACTS, "periodic_skips") if cfg[t]["enabled"]]
    step = state_step(states) if state_tests else None
    if state_tests and step is None:
        skips += [_skip(CATEGORY, t, ALL_CAMERAS, "no_robot_state") for t in state_tests]
    if step is not None:
        moving = robot_moving(step, moving_fraction)
        if cfg["periodic_skips"]["enabled"]:
            add("periodic_skips", ALL_CAMERAS, check_periodic_skips(step, moving, control_freq, **_params(cfg["periodic_skips"])))

    durations: dict[str, float] = {}
    for camera, path in sorted(videos.items()):
        if not Path(path).is_file():
            if cfg["decoding"]["enabled"]:
                add("decoding", camera, [f"missing file {Path(path).name}"])
            continue
        p = probe_video(Path(path))
        if cfg["resolution"]["enabled"]:
            add("resolution", camera, check_resolution(p, **_params(cfg["resolution"])))
        if cfg["decoding"]["enabled"]:
            add("decoding", camera, check_decoding(p, **_params(cfg["decoding"])))
        if cfg["brightness"]["enabled"]:
            add("brightness", camera, check_brightness(p, **_params(cfg["brightness"])))
        if cfg["speed"]["enabled"]:
            add("speed", camera, check_speed(p, n_steps, control_freq, **_params(cfg["speed"])))
        if cfg[ARTIFACTS]["enabled"]:
            rate = p.fps or control_freq or 0.0
            add(ARTIFACTS, camera, check_frames(p, moving, rate, **artifacts_params))
        if p.duration is not None:
            durations[camera] = p.duration

    if cfg["speed"]["enabled"] and len(durations) > 1:
        spread = max(durations.values()) - min(durations.values())
        if spread > cfg["speed"]["duration_tol_s"]:
            detail = ", ".join(f"{c} {d:.1f}s" for c, d in durations.items())
            add("speed", ALL_CAMERAS, [f"camera durations differ: {detail}"])

    return {"findings": findings, "skips": skips, "checked": [(CATEGORY, t) for t in enabled]}
