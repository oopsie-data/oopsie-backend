"""
Convert RoboArena evaluation sessions to oopsiedata_format_v1 HDF5.

Source structure (one session):
  {session_dir}/
    metadata.yaml                      — session-level metadata
    {letter}_{policy_name}/            — one folder per evaluated policy
      {policy_name}_{ts}_npz_file.npz  — per-step trajectory dicts
      {policy_name}_{ts}_video_left.mp4
      {policy_name}_{ts}_video_right.mp4
      {policy_name}_{ts}_video_wrist.mp4

metadata.yaml keys used:
  language_instruction  → HDF5 root attr
  evaluation_location   → lab_id root attr
  evaluator_name        → operator_name root attr
  policies.{letter}.policy_name     → robot_profile.policy_name
  policies.{letter}.binary_success  → episode_annotations success

NPZ data array: list of per-step dicts with:
  cartesian_position  (6,)  — xyz + euler_xyz
  joint_position      (7,)
  gripper_position    (1,)
  action              (8,)  — first 7: joint velocities, last 1: gripper position

Output (one file per policy per session):
  {output_dir}/
    {episode_id}.h5
    videos/{episode_id}_left.mp4
    videos/{episode_id}_right.mp4
    videos/{episode_id}_wrist.mp4

Usage:
    # Convert a single session
    python -m oopsie_backend.conversion.convert_roboarena_data \\
        -s /datastor2/droid/roboarena_2026_02/evaluation_sessions/fff333cb-... \\
        -o /path/to/output

    # Convert all sessions under a parent directory
    python -m oopsie_backend.conversion.convert_roboarena_data \\
        -s /datastor2/droid/roboarena_2026_02/evaluation_sessions \\
        -o /path/to/output

    # Start episode IDs from a specific offset
    python -m oopsie_backend.conversion.convert_roboarena_data \\
        -s /datastor2/droid/roboarena_2026_02/evaluation_sessions \\
        -o /path/to/output --start-id 500
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    robot_profile_to_json,
)

from oopsie_backend.conversion.conversion_utils import SCHEMA_VERSION
CONTROL_FREQ = 10  # Hz — matches source MP4 frame rate
CAMERA_NAMES = ["left", "right", "wrist"]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

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
_ACTIVE_ACTION_SPACE = ["joint_velocity", "gripper_position"]


def _make_robot_profile(policy_name: str, camera_names: list[str]) -> RobotProfile:
    return RobotProfile(
        policy_name=policy_name,
        robot_name="franka_research_3",
        is_biarm=False,
        uses_mobile_base=False,
        gripper_name="robotiq_2f_85",
        control_freq=CONTROL_FREQ,
        camera_names=camera_names,
        robot_state_keys=["joint_position", "gripper_position", "cartesian_position"],
        robot_state_joint_names=list(JOINT_NAMES),
        robot_state_orientation_representation="euler_xyz",
        action_space=list(_ACTIVE_ACTION_SPACE),
        action_joint_names=list(JOINT_NAMES),
    )


def _load_npz_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    """Stack per-step dicts from the NPZ data array into (T, D) float64 arrays."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    T = len(data)

    joint_pos = np.empty((T, 7), dtype=np.float64)
    gripper_pos = np.empty((T, 1), dtype=np.float64)
    cartesian_pos = np.empty((T, 6), dtype=np.float64)
    joint_vel_act = np.empty((T, 7), dtype=np.float64)
    gripper_act = np.empty((T, 1), dtype=np.float64)

    for t, step in enumerate(data):
        joint_pos[t] = step["joint_position"]
        gripper_pos[t, 0] = step["gripper_position"][0]
        cartesian_pos[t] = step["cartesian_position"]
        action = step["action"]
        joint_vel_act[t] = action[:7]
        gripper_act[t, 0] = action[7]

    return {
        "joint_position": joint_pos,
        "gripper_position": gripper_pos,
        "cartesian_position": cartesian_pos,
        "joint_velocity": joint_vel_act,
        "gripper_act": gripper_act,
    }


def _find_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[0]


def convert_policy_episode(
    session_dir: Path,
    policy_letter: str,
    policy_info: dict,
    language_instruction: str,
    evaluator_name: str,
    evaluation_location: str,
    output_dir: Path,
    episode_id: str,
) -> Path:
    """Convert one policy episode from a roboarena session to an HDF5 file.

    Returns the path to the written HDF5 file.
    """
    policy_name = policy_info["policy_name"]
    binary_success = float(policy_info["binary_success"])

    policy_dir = session_dir / f"{policy_letter}_{policy_name}"
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    npz_path = _find_file(policy_dir, "*_npz_file.npz")

    # Detect which cameras are actually present — not all episodes have all three.
    video_src: dict[str, Path] = {}
    for cam in CAMERA_NAMES:
        matches = list(policy_dir.glob(f"*_video_{cam}.mp4"))
        if matches:
            video_src[cam] = matches[0]
    present_cameras = list(video_src.keys())
    if not present_cameras:
        raise FileNotFoundError(f"No camera videos found in {policy_dir}")

    output_h5 = output_dir / f"{episode_id}.h5"
    if output_h5.exists():
        tqdm.write(f"  Skipping {output_h5.name} (already exists)")
        return output_h5

    arrays = _load_npz_arrays(npz_path)
    T = arrays["joint_position"].shape[0]

    # Copy videos — symlinks break on upload
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    rel_video_paths: dict[str, str] = {}
    for cam in present_cameras:
        dst = videos_dir / f"{episode_id}_{cam}.mp4"
        shutil.copy2(video_src[cam], dst)
        rel_video_paths[cam] = f"videos/{episode_id}_{cam}.mp4"

    profile = _make_robot_profile(policy_name, present_cameras)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_h5, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["episode_id"] = episode_id
        f.attrs["language_instruction"] = language_instruction
        f.attrs["lab_id"] = evaluation_location
        f.attrs["operator_name"] = evaluator_name
        f.attrs.create("robot_profile", robot_profile_to_json(profile), dtype=str_dtype)

        obs = f.create_group("observations")

        vp = obs.create_group("video_paths")
        for cam in present_cameras:
            vp.create_dataset(cam, data=rel_video_paths[cam], dtype=str_dtype)

        rs = obs.create_group("robot_states")
        rs.create_dataset("joint_position", data=arrays["joint_position"])
        rs.create_dataset("gripper_position", data=arrays["gripper_position"])
        rs.create_dataset("cartesian_position", data=arrays["cartesian_position"])

        acts = f.create_group("actions")
        for key in _ALL_ACTION_KEYS:
            if key == "joint_velocity":
                acts.create_dataset(key, data=arrays["joint_velocity"])
            elif key == "gripper_position":
                acts.create_dataset(key, data=arrays["gripper_act"])
            else:
                acts.create_dataset(key, data=h5py.Empty(dtype=np.float64))

        ea = f.create_group("episode_annotations")
        ann = ea.create_group("roboarena_eval")
        ann.attrs["success"] = np.float64(binary_success)

    tqdm.write(f"  Wrote {output_h5.name} (T={T}, success={binary_success})")
    return output_h5


def convert_session(
    session_dir: Path,
    output_dir: Path,
    start_id: int = 0,
) -> tuple[list[Path], int]:
    """Convert all policy episodes in one session.

    Returns (list of written HDF5 paths, next available episode id).
    """
    metadata_path = session_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.yaml not found in {session_dir}")

    with metadata_path.open() as fh:
        meta = yaml.safe_load(fh)

    language_instruction: str = meta["language_instruction"]
    evaluator_name: str = meta["evaluator_name"]
    evaluation_location: str = meta["evaluation_location"]
    policies: dict = meta["policies"]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    episode_counter = start_id
    letters = sorted(policies)

    with tqdm(letters, desc=session_dir.name, unit="ep", leave=False) as pbar:
        for letter in pbar:
            policy_info = policies[letter]
            episode_id = f"episode_{episode_counter:06d}"
            pbar.set_postfix_str(f"{policy_info['policy_name']} → {episode_id}")
            try:
                h5_path = convert_policy_episode(
                    session_dir=session_dir,
                    policy_letter=letter,
                    policy_info=policy_info,
                    language_instruction=language_instruction,
                    evaluator_name=evaluator_name,
                    evaluation_location=evaluation_location,
                    output_dir=output_dir,
                    episode_id=episode_id,
                )
                written.append(h5_path)
            except Exception as exc:
                tqdm.write(f"  ✗ [{letter}] {policy_info['policy_name']}: {exc}")
            episode_counter += 1

    return written, episode_counter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RoboArena evaluation sessions to oopsiedata_format_v1 HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        type=Path,
        help=(
            "Path to a single session directory (containing metadata.yaml) "
            "or a parent directory of multiple session directories."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        type=Path,
        help="Output directory for HDF5 files and copied videos.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting numeric suffix for episode IDs (default: 0).",
    )
    args = parser.parse_args()

    source: Path = args.source.resolve()
    output_dir: Path = args.output_dir.resolve()

    # Determine whether source is a single session or a parent of sessions.
    if (source / "metadata.yaml").is_file():
        # Single session
        session_dirs = [source]
    else:
        session_dirs = sorted(
            d for d in source.iterdir()
            if d.is_dir() and (d / "metadata.yaml").is_file()
        )
        if not session_dirs:
            parser.error(
                f"No session directories with metadata.yaml found under {source}"
            )

    total_written = 0
    total_failed = 0
    episode_counter = args.start_id

    with tqdm(session_dirs, desc="Sessions", unit="session") as pbar:
        for session_dir in pbar:
            pbar.set_postfix_str(session_dir.name[:40])
            try:
                written, episode_counter = convert_session(
                    session_dir=session_dir,
                    output_dir=output_dir,
                    start_id=episode_counter,
                )
                total_written += len(written)
            except Exception as exc:
                tqdm.write(f"✗ Session {session_dir.name}: {exc}")
                total_failed += 1

    tqdm.write(f"\nDone: {total_written} episode(s) written, {total_failed} session failure(s).")


if __name__ == "__main__":
    main()
