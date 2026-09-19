"""
Convert the SOAR self-improvement dataset to oopsiedata_format_v1 HDF5.

SOAR (https://auto-improvement.github.io) is a large WidowX / BridgeData-V2
dataset of autonomously-collected trajectories, ~68% of which are failures
(auto-labeled by a VLM). The local copy is laid out as:

  {source}/berkeley_robot_{0..4}/{task}/{policy}/{date}/{success|failure}/traj{N}/
    actions.npy        (T, 7) float64  [dx,dy,dz,droll,dpitch,dyaw, gripper∈{0,1}]  (arm = DELTA EEF command)
    eef_poses.npy      (T, 7) float64  [x,y,z,roll,pitch,yaw, gripper]              (absolute proprio)
    trajectory.mp4     256x256, 15 fps container, T frames (one per 5 Hz control step)
    goals.mp4 / combined.mp4   VLM goal images / viz  (NOT robot cameras — ignored)
    language_task.txt  natural-language instruction
    success.txt        "True" / "False"
    robot_id.txt       "0".."4"
    time.txt, task_list.txt, object_list.txt  (unused)

The SOURCE directory is treated as strictly READ-ONLY: this script only
np.load / open(read) from it and never writes into it.

Video fps note: source MP4s are packaged at SOURCE_VIDEO_FPS (15 fps) but the robot
control loop runs at control_freq (5 Hz), so there is one video frame per control step.
The oopsie validator requires |frame_count/fps - T/control_freq| ≤ 0.5s, so we
re-encode each video at control_freq fps (stretching PTS by SOURCE_VIDEO_FPS/control_freq
= 3x) to make the container duration match the true episode duration.

Action representation (per dataset-owner decision):
  The oopsie action_space is `cartesian_position` (absolute). SOAR stores delta
  arm commands plus absolute proprio, so we reconstruct the absolute action as
  the NEXT proprio state:
    actions/cartesian_position[t] = eef_poses[t+1, :6]   (last step: eef_poses[-1,:6] + actions[-1,:6])
    actions/gripper_position[t]   = eef_poses[t+1, 6]    (last step: eef_poses[-1, 6])

joint_position observation is a zeros placeholder: SOAR has no joint data, but
the oopsie tooling requires joint_position in robot_state_keys.

Output (one file per trajectory):
  {output_dir}/
    {episode_id}.h5
    videos/{episode_id}_main.mp4

Usage:
    # Small sanity batch
    python -m oopsie_backend.external_conversion.convert_soar_data \\
        -o /tmp/soar_out --max-episodes 5

    # Full conversion
    python -m oopsie_backend.external_conversion.convert_soar_data \\
        -s /home/paulzhou/soar-dataset-local -o /path/to/output

    # Failures only, robot 1 only
    python -m oopsie_backend.external_conversion.convert_soar_data \\
        -o /path/to/output --outcome failure --robot-id 1
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    load_robot_profile,
    robot_profile_to_json,
)

from oopsie_backend.external_conversion.conversion_utils import SCHEMA_VERSION

DEFAULT_SOURCE = Path("/home/paulzhou/soar-dataset-local")
DEFAULT_ROBOT_PROFILE = (
    Path(__file__).resolve().parents[2] / "configs" / "robot_profiles" / "widow_x_profile.yaml"
)
DEFAULT_LAB_ID = "Berkeley_RAIL"
DEFAULT_OPERATOR_NAME = "SOAR_policy"
DEFAULT_ANNOTATOR_NAME = "soar_vlm"

# Episode-duration bounds (seconds) enforced by the validator.
MIN_EPISODE_SECONDS = 1
MAX_EPISODE_SECONDS = 300

# Source videos are packaged at this fps in the MP4 container, but the robot
# control loop runs at control_freq (5 Hz) — one frame per control step.
SOURCE_VIDEO_FPS = 15

# Write h5py.Empty for every key not in the active action space.
_ALL_ACTION_KEYS = [
    "cartesian_position",
    "cartesian_velocity",
    "joint_position",
    "joint_velocity",
    "base_position",
    "base_velocity",
    "gripper_velocity",
    "gripper_position",
    "gripper_binary",
]


def _transcode_video(src: Path, dst: Path, control_freq: float) -> None:
    """Re-encode src MP4 at control_freq fps, preserving all frames.

    Source videos are packaged at SOURCE_VIDEO_FPS but the robot runs at
    control_freq, so PTS are stretched by SOURCE_VIDEO_FPS/control_freq.
    This makes the container duration == T/control_freq seconds, which the
    oopsie validator requires.
    """
    if abs(SOURCE_VIDEO_FPS - control_freq) < 0.01:
        shutil.copy2(src, dst)
        return
    multiplier = SOURCE_VIDEO_FPS / control_freq
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"setpts={multiplier:.6f}*PTS",
            "-r", str(int(control_freq)),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def _sanitize(text: str) -> str:
    """Filesystem/identifier-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "x"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _build_actions(eef_poses: np.ndarray, actions_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute cartesian_position + gripper action = next proprio state.

    cartesian_position[t] = eef_poses[t+1, :6]; last step = eef_poses[-1,:6] + delta[-1,:6]
    gripper_position[t]   = eef_poses[t+1, 6];  last step = eef_poses[-1, 6]
    """
    T = eef_poses.shape[0]
    cart = np.empty((T, 6), dtype=np.float64)
    grip = np.empty((T, 1), dtype=np.float64)

    if T > 1:
        cart[:-1] = eef_poses[1:, :6]
        grip[:-1, 0] = eef_poses[1:, 6]
    cart[-1] = eef_poses[-1, :6] + actions_raw[-1, :6]
    grip[-1, 0] = eef_poses[-1, 6]
    return cart, grip


def _episode_id(rel_parts: tuple[str, ...], counter: int) -> str:
    """Globally-unique, descriptive episode id from the relative path components."""
    slug = "_".join(_sanitize(p) for p in rel_parts)
    return f"soar_{counter:06d}_{slug}"


def convert_trajectory(
    traj_dir: Path,
    source: Path,
    base_profile: RobotProfile,
    output_dir: Path,
    videos_dir: Path,
    episode_id: str,
    lab_id: str,
    operator_name: str,
    annotator_name: str,
    overwrite: bool,
) -> str | None:
    """Convert one SOAR trajectory to an HDF5 file.

    Returns a status string ("written" / "skipped" / "exists"), or None if the
    trajectory was filtered out by duration bounds.
    """
    output_h5 = output_dir / f"{episode_id}.h5"
    if output_h5.exists() and not overwrite:
        return "exists"

    actions_raw = np.load(traj_dir / "actions.npy")
    eef_poses = np.load(traj_dir / "eef_poses.npy")
    T = eef_poses.shape[0]

    control_freq = base_profile.control_freq
    duration = T / control_freq
    if not (MIN_EPISODE_SECONDS <= duration <= MAX_EPISODE_SECONDS):
        tqdm.write(f"  skip {episode_id}: T={T} -> {duration:.2f}s out of [1, 300]s")
        return None

    language_instruction = _read_text(traj_dir / "language_task.txt")
    success = 1.0 if _read_text(traj_dir / "success.txt").lower() == "true" else 0.0

    # Policy name comes from the path: .../{task}/{policy}/{date}/{outcome}/{trajN}
    rel = traj_dir.relative_to(source).parts
    policy_name = rel[2] if len(rel) >= 5 else base_profile.policy_name
    profile = dataclasses.replace(base_profile, policy_name=policy_name)

    # Robot states
    cartesian_obs = np.ascontiguousarray(eef_poses[:, :6], dtype=np.float64)
    gripper_obs = np.ascontiguousarray(eef_poses[:, 6:7], dtype=np.float64)
    joint_obs = np.zeros((T, len(profile.robot_state_joint_names)), dtype=np.float64)

    # Actions (absolute, reconstructed from proprio)
    cartesian_act, gripper_act = _build_actions(eef_poses, actions_raw)

    # Re-encode video at control_freq fps so container duration == T/control_freq.
    src_video = traj_dir / "trajectory.mp4"
    if not src_video.is_file():
        tqdm.write(f"  skip {episode_id}: missing trajectory.mp4")
        return None
    dst_video = videos_dir / f"{episode_id}_main.mp4"
    _transcode_video(src_video, dst_video, float(control_freq))
    rel_video_path = f"videos/{episode_id}_main.mp4"

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_h5, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["episode_id"] = episode_id
        f.attrs["language_instruction"] = language_instruction
        f.attrs["lab_id"] = lab_id
        f.attrs["operator_name"] = operator_name
        f.attrs.create("robot_profile", robot_profile_to_json(profile), dtype=str_dtype)

        obs = f.create_group("observations")

        vp = obs.create_group("video_paths")
        vp.create_dataset("main", data=rel_video_path, dtype=str_dtype)

        rs = obs.create_group("robot_states")
        rs.create_dataset("joint_position", data=joint_obs)
        rs.create_dataset("gripper_position", data=gripper_obs)
        rs.create_dataset("cartesian_position", data=cartesian_obs)

        acts = f.create_group("actions")
        for key in _ALL_ACTION_KEYS:
            if key == "cartesian_position":
                acts.create_dataset(key, data=cartesian_act)
            elif key == "gripper_position":
                acts.create_dataset(key, data=gripper_act)
            else:
                acts.create_dataset(key, data=h5py.Empty(dtype=np.float64))

        ea = f.create_group("episode_annotations")
        ann = ea.create_group(annotator_name)
        ann.attrs["success"] = np.float64(success)

    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the SOAR dataset to oopsiedata_format_v1 HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", "-s", type=Path, default=DEFAULT_SOURCE,
                        help=f"SOAR dataset root (default: {DEFAULT_SOURCE}).")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for HDF5 files and copied videos.")
    parser.add_argument("--robot-profile", type=Path, default=DEFAULT_ROBOT_PROFILE,
                        help=f"Robot profile YAML (default: {DEFAULT_ROBOT_PROFILE}).")
    parser.add_argument("--lab-id", default=DEFAULT_LAB_ID)
    parser.add_argument("--operator-name", default=DEFAULT_OPERATOR_NAME)
    parser.add_argument("--annotator-name", default=DEFAULT_ANNOTATOR_NAME)
    parser.add_argument("--outcome", choices=["all", "success", "failure"], default="all",
                        help="Filter by VLM success label (default: all).")
    parser.add_argument("--robot-id", default=None,
                        help="Only convert berkeley_robot_<id> (e.g. 1). Default: all robots.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Stop after writing this many episodes.")
    parser.add_argument("--start-id", type=int, default=0,
                        help="Starting numeric counter for episode IDs (default: 0).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output HDF5 files.")
    args = parser.parse_args()

    source: Path = args.source.resolve()
    output_dir: Path = args.output_dir.resolve()
    if not source.is_dir():
        parser.error(f"Source directory not found: {source}")

    base_profile = load_robot_profile(args.robot_profile)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Discover trajectories (dir containing actions.npy), deterministically ordered.
    traj_dirs = sorted(p.parent for p in source.rglob("actions.npy"))

    # Apply path-based filters.
    if args.robot_id is not None:
        robot_dir = f"berkeley_robot_{args.robot_id}"
        traj_dirs = [d for d in traj_dirs if robot_dir in d.relative_to(source).parts]
    if args.outcome != "all":
        traj_dirs = [d for d in traj_dirs if args.outcome in d.relative_to(source).parts]

    counts = {"written": 0, "exists": 0, "skipped": 0, "failed": 0}
    counter = args.start_id

    with tqdm(traj_dirs, desc="SOAR", unit="traj") as pbar:
        for traj_dir in pbar:
            if args.max_episodes is not None and counts["written"] >= args.max_episodes:
                break
            rel_parts = traj_dir.relative_to(source).parts
            episode_id = _episode_id(rel_parts, counter)
            pbar.set_postfix_str(episode_id[:48])
            try:
                status = convert_trajectory(
                    traj_dir=traj_dir,
                    source=source,
                    base_profile=base_profile,
                    output_dir=output_dir,
                    videos_dir=videos_dir,
                    episode_id=episode_id,
                    lab_id=args.lab_id,
                    operator_name=args.operator_name,
                    annotator_name=args.annotator_name,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch going
                tqdm.write(f"  ✗ {episode_id}: {exc}")
                counts["failed"] += 1
                counter += 1
                continue

            if status is None:
                counts["skipped"] += 1
            else:
                counts[status] += 1
            counter += 1

    tqdm.write(
        f"\nDone: {counts['written']} written, {counts['exists']} already existed, "
        f"{counts['skipped']} skipped (filters/duration), {counts['failed']} failed."
    )


if __name__ == "__main__":
    main()
