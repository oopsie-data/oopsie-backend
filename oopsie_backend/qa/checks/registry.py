"""Every QA check, imported here by name so its ``@check`` registers it.

To add a check, write ``qa/checks/<category>/<test>.py`` with an ``@check``
function, give it a section in ``configs/action_qa_config.yaml`` and import it
below. To remove one, delete its import and file.
"""

from __future__ import annotations

import oopsie_backend.qa.checks.cartesian_position.biarm_ordering  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.delta_detection  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.normalized_range  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.pose_tracking  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.position_units  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.quaternion_scalar_order  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.quaternion_shape  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.quaternion_sign_flip  # noqa: F401
import oopsie_backend.qa.checks.cartesian_position.rotation_jump  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.linear_angular_order  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.looks_like_position  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.normalized_range  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.shape  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.smoothness  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.speed_limit  # noqa: F401
import oopsie_backend.qa.checks.cartesian_velocity.tracking  # noqa: F401
import oopsie_backend.qa.checks.general.action_chunk_shape  # noqa: F401
import oopsie_backend.qa.checks.general.action_copies_state  # noqa: F401
import oopsie_backend.qa.checks.general.command_rate  # noqa: F401
import oopsie_backend.qa.checks.general.constant_action  # noqa: F401
import oopsie_backend.qa.checks.general.delta_detection  # noqa: F401
import oopsie_backend.qa.checks.general.duplicate_pairs  # noqa: F401
import oopsie_backend.qa.checks.general.gripper_binary_present  # noqa: F401
import oopsie_backend.qa.checks.general.gripper_command_range  # noqa: F401
import oopsie_backend.qa.checks.general.gripper_range  # noqa: F401
import oopsie_backend.qa.checks.general.length_mismatch  # noqa: F401
import oopsie_backend.qa.checks.general.placeholder_metadata  # noqa: F401
import oopsie_backend.qa.checks.general.required_state_keys  # noqa: F401
import oopsie_backend.qa.checks.general.scale_consistency  # noqa: F401
import oopsie_backend.qa.checks.general.static_robot  # noqa: F401
import oopsie_backend.qa.checks.general.tracking_residual  # noqa: F401
import oopsie_backend.qa.checks.joint_position.delta_detection  # noqa: F401
import oopsie_backend.qa.checks.joint_position.dof  # noqa: F401
import oopsie_backend.qa.checks.joint_position.gripper_in_joints  # noqa: F401
import oopsie_backend.qa.checks.joint_position.joint_names  # noqa: F401
import oopsie_backend.qa.checks.joint_position.normalized_range  # noqa: F401
import oopsie_backend.qa.checks.joint_position.radians  # noqa: F401
import oopsie_backend.qa.checks.joint_position.step_jump  # noqa: F401
import oopsie_backend.qa.checks.joint_position.tracking  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.dof  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.looks_like_position  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.normalized_range  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.smoothness  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.speed_limit  # noqa: F401
import oopsie_backend.qa.checks.joint_velocity.tracking  # noqa: F401
import oopsie_backend.qa.checks.video.brightness  # noqa: F401
import oopsie_backend.qa.checks.video.browser_playable  # noqa: F401
import oopsie_backend.qa.checks.video.camera_duration_mismatch  # noqa: F401
import oopsie_backend.qa.checks.video.choppy  # noqa: F401
import oopsie_backend.qa.checks.video.corrupted_frames  # noqa: F401
import oopsie_backend.qa.checks.video.decoding  # noqa: F401
import oopsie_backend.qa.checks.video.duration  # noqa: F401
import oopsie_backend.qa.checks.video.fps  # noqa: F401
import oopsie_backend.qa.checks.video.frozen  # noqa: F401
import oopsie_backend.qa.checks.video.periodic_skips  # noqa: F401
import oopsie_backend.qa.checks.video.recording_gap  # noqa: F401
import oopsie_backend.qa.checks.video.resolution  # noqa: F401
import oopsie_backend.qa.checks.video.state_jump  # noqa: F401
import oopsie_backend.qa.checks.video.video_jump  # noqa: F401
from oopsie_backend.qa.checks.base import CHECKS, Check


def all_checks() -> dict[str, Check]:
    """Check name (``category/test``) -> check."""
    return dict(CHECKS)


def enabled_checks(config: dict, only: list[str] | None = None) -> dict[str, Check]:
    """Checks enabled in ``config``, optionally restricted to the names in ``only``."""
    unconfigured = sorted(n for n, c in CHECKS.items() if c.test not in config.get(c.category, {}))
    if unconfigured:
        raise ValueError(f"checks without a config section: {unconfigured}")
    if only:
        unknown = sorted(set(only) - set(CHECKS))
        if unknown:
            raise ValueError(f"unknown checks: {unknown}")
    return {n: c for n, c in CHECKS.items() if c.enabled(config) and (not only or n in only)}
