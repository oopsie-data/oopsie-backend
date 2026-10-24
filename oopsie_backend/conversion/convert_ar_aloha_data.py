"""
Convert ACT/ALOHA format HDF5 to the robotic failure dataset format.

Source format datasets:
  /action                                        (T, 14)  joint actions, both arms
  /success                                       (T,)     per-timestep success (aggregated to scalar)
  /compress_len                                  (3, T)   actual JPEG byte lengths per camera
  /observations/images/camera_high               (T, max_bytes)  JPEG-compressed frames
  /observations/images/camera_wrist_left         (T, max_bytes)  JPEG-compressed frames
  /observations/images/camera_wrist_right        (T, max_bytes)  JPEG-compressed frames
  /observations/qpos                             (T, 14)  joint positions, both arms
  /observations/qvel                             (T, 14)  joint velocities (optional)
  /observations/effort                           (T, 14)  joint torques (optional)

Target format (robotic failure dataset):
  action_dict/cartesian_position           [zeros — FK not available]
  action_dict/cartesian_velocity           [zeros — FK not available]
  action_dict/gripper_position        (T, 2)   from action[:, GRIPPER_IDX]
  action_dict/gripper_velocity        (T, 2)   zeros
  image_observations/exterior_image_1_left  scalar str → mp4 path
  image_observations/exterior_image_2_left  scalar str → mp4 path
  image_observations/wrist_image_left       scalar str → mp4 path
  episode_annotations/episode_id      scalar str
  episode_annotations/success         scalar float
  episode_annotations/failure_category     scalar str
  episode_annotations/failure_description  scalar str
  episode_annotations/would_retry     scalar float
  episode_annotations/lab_id          scalar str
  episode_annotations/operator_name   scalar str
  episode_annotations/policy_id       scalar str
  episode_annotations/robot_id        scalar str
  episode_annotations/control_freq    scalar str
  observation/cartesian_position           [zeros — FK not available]
  observation/gripper_position        (T, 2)   from qpos[:, GRIPPER_IDX]
  observation/joint_position          (T, 12)  from qpos[:, ARM_IDX]
  [file attr] language_instruction    scalar str  (from --language CLI arg)

Usage:
    python convert_ar_aloha_data.py -s /path/to/episode_0.hdf5 -o /path/to/output --language "Pick up the ball"
    python convert_ar_aloha_data.py -s /path/to/episode_0.hdf5 -o /path/to/output --language "task" -e 000001
    python convert_ar_aloha_data.py -s /path/to/rollouts/ -o /path/to/output --language "task"
    python convert_ar_aloha_data.py -s /path/to/rollouts/ -o /path/to/output --language "task" --start-id 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from oopsie_data_tools.annotation_tool.episode_recorder import write_mp4

from oopsie_backend.conversion.conversion_utils import (
    SCHEMA_VERSION,
    _decode_text,
    _parse_fps,
    _resize_frames,
    write_episode_annotations,
)


DEFAULT_CONTROL_FREQ = "50"

# Indices within the 14D ALOHA qpos/action vectors
ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 12 arm joints
GRIPPER_IDX = [6, 13]                                  # 2 gripper joints

CAMERA_MAP = [
    ("camera_high",        "cam_high"),
    ("camera_wrist_right", "cam_right_wrist"),
    ("camera_wrist_left",  "cam_left_wrist"),
]


def _decode_jpeg_frames(raw: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Decode T JPEG frames from a padded uint8 array into (T, H, W, 3) RGB."""
    frames = []
    for t in range(raw.shape[0]):
        n = int(lengths[t])
        buf = np.frombuffer(raw[t, :n], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode JPEG frame {t}")
        frames.append(bgr[:, :, ::-1])  # BGR → RGB
    return np.stack(frames, axis=0)


def convert(
    source_h5: str,
    output_dir: Path,
    language_instruction: str,
    episode_id: str = "000000",
    control_freq: str = DEFAULT_CONTROL_FREQ,
    lab_id: str = "",
    operator_name: str = "",
    policy_id: str = "",
    robot_id: str = "",
    success_aggregation: str = "max",
) -> Path:
    """
    Convert a single ACT/ALOHA HDF5 file to the robotic failure dataset format.

    Returns:
        Path to the output HDF5 file.
    """
    episode_id = str(episode_id).zfill(6)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / f"{episode_id}_trajectory.h5"
    str_dtype = h5py.string_dtype(encoding="utf-8")
    video_fps = _parse_fps(control_freq)

    print(f"\nReading source: {source_h5}")
    with h5py.File(source_h5, "r") as src:
        # Trajectory data
        action = src["action"][()]        # (T, 14)
        qpos = src["observations/qpos"][()]  # (T, 14)

        # Per-timestep success → episode scalar
        success_arr = src["success"][()]  # (T,)
        if success_aggregation == "last":
            success = float(success_arr[-1])
        else:
            success = float(success_arr.max())

        # JPEG-compressed camera images
        compress_len = src["compress_len"][()]  # (3, T)
        cameras: list[np.ndarray] = []
        for cam_idx, (src_key, _) in enumerate(CAMERA_MAP):
            raw = src[f"observations/images/{src_key}"][()]  # (T, max_bytes)
            cameras.append(_decode_jpeg_frames(raw, compress_len[cam_idx]))

    cam_high, cam_rw, cam_lw = cameras  # matches CAMERA_MAP order

    T = action.shape[0]
    print(f"Trajectory length: {T} steps")
    print(f"Language instruction: {language_instruction!r}")
    print(f"Success: {success}")

    # ── Video encoding ──────────────────────────────────────────────────────────
    # camera_high        → exterior_image_1_left
    # camera_wrist_right → exterior_image_2_left
    # camera_wrist_left  → wrist_image_left
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    ext1_rel = f"videos/{episode_id}_cam_high.mp4"
    ext2_rel = f"videos/{episode_id}_cam_right_wrist.mp4"
    wrist_rel = f"videos/{episode_id}_cam_left_wrist.mp4"

    print("\nEncoding videos (this may take a few minutes)...")
    write_mp4(output_dir / ext1_rel, _resize_frames(cam_high), video_fps)
    write_mp4(output_dir / ext2_rel, _resize_frames(cam_rw), video_fps)
    write_mp4(output_dir / wrist_rel, _resize_frames(cam_lw), video_fps)

    # ── Observation fields ──────────────────────────────────────────────────────
    gripper_pos_obs = qpos[:, GRIPPER_IDX].astype(np.float32)  # (T, 2)

    # ── Write output HDF5 ───────────────────────────────────────────────────────
    print(f"\nWriting: {output_h5_path}")
    with h5py.File(output_h5_path, "w") as dst:
        dst.attrs["schema"] = SCHEMA_VERSION

        write_episode_annotations(
            file_handle=dst,
            episode_id=episode_id,
            success=success,
            lab_id=lab_id,
            operator_name=operator_name,
            policy_id=policy_id,
            robot_id=robot_id,
            control_freq=control_freq,
        )

        img = dst.create_group("image_observations")
        img.create_dataset("exterior_image_1_left", data=ext1_rel, dtype=str_dtype)
        img.create_dataset("exterior_image_2_left", data=ext2_rel, dtype=str_dtype)
        img.create_dataset("wrist_image_left", data=wrist_rel, dtype=str_dtype)

        ad = dst.create_group("action_dict")
        ad.create_dataset(
            "cartesian_position", data=np.zeros((T, 6), dtype=np.float32)
        )
        ad.create_dataset(
            "cartesian_velocity", data=np.zeros((T, 6), dtype=np.float32)
        )
        ad.create_dataset("joint_position", data=action[:, ARM_IDX].astype(np.float32))
        ad.create_dataset(
            "gripper_position", data=action[:, GRIPPER_IDX].astype(np.float32)
        )
        ad.create_dataset("gripper_velocity", data=np.zeros((T, 2), dtype=np.float32))

        obs = dst.create_group("observation")
        obs.create_dataset(
            "cartesian_position", data=np.zeros((T, 6), dtype=np.float32)
        )
        obs.create_dataset("gripper_position", data=gripper_pos_obs)
        obs.create_dataset("joint_position", data=qpos[:, ARM_IDX].astype(np.float32))

        dst.attrs["language_instruction"] = language_instruction

    print(f"\nDone. Wrote {output_h5_path}")
    return output_h5_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACT/ALOHA HDF5 to robotic failure dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help=(
            "Path to a single source HDF5 file (e.g. episode_0.hdf5) "
            "or a directory of HDF5 files to convert in bulk."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--episode-id",
        "-e",
        default=None,
        help=(
            "Output episode ID (zero-padded to 6 digits, e.g. 000001). "
            "Only valid when --source is a single file. "
            "When converting a directory, IDs are assigned automatically starting from 0."
        ),
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting episode ID when converting a directory (default: 0).",
    )
    parser.add_argument(
        "--control-freq",
        type=str,
        default=DEFAULT_CONTROL_FREQ,
        help=f"Control frequency in Hz, used as video FPS (default: {DEFAULT_CONTROL_FREQ})",
    )
    parser.add_argument(
        "--lab-id", type=str, default="", help="Value for episode_annotations/lab_id."
    )
    parser.add_argument(
        "--operator-name",
        type=str,
        default="",
        help="Value for episode_annotations/operator_name.",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="",
        help="Value for episode_annotations/policy_id.",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="",
        help="Value for episode_annotations/robot_id.",
    )
    parser.add_argument(
        "--language",
        "-l",
        required=True,
        help="Language instruction for all episodes (not stored in source files).",
    )
    parser.add_argument(
        "--success-aggregation",
        choices=["max", "last"],
        default="max",
        help="How to collapse per-timestep success to a scalar: 'max' or 'last' (default: max).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    shared_kwargs = dict(
        output_dir=args.output_dir,
        language_instruction=args.language,
        control_freq=args.control_freq,
        lab_id=args.lab_id,
        operator_name=args.operator_name,
        policy_id=args.policy_id,
        robot_id=args.robot_id,
        success_aggregation=args.success_aggregation,
    )

    if source.is_dir():
        if args.episode_id is not None:
            parser.error(
                "--episode-id cannot be used with a directory source; use --start-id instead."
            )
        hdf5_files = sorted(source.glob("*.hdf5")) + sorted(source.glob("*.h5"))
        if not hdf5_files:
            parser.error(f"No .hdf5 or .h5 files found in {source}")
        print(f"\nFound {len(hdf5_files)} file(s) in {source}")
        failures = 0
        for i, h5_file in enumerate(hdf5_files):
            episode_id = str(args.start_id + i).zfill(6)
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(hdf5_files)}] {h5_file.name} → episode {episode_id}")
            print(f"{'=' * 60}")
            try:
                convert(source_h5=str(h5_file), episode_id=episode_id, **shared_kwargs)
            except Exception as e:
                print(f"\n✗ Failed: {e}")
                failures += 1
        passed = len(hdf5_files) - failures
        print(f"\nDone: {passed}/{len(hdf5_files)} converted successfully.")
    else:
        episode_id = args.episode_id if args.episode_id is not None else "000000"
        convert(source_h5=str(source), episode_id=episode_id, **shared_kwargs)


if __name__ == "__main__":
    main()
