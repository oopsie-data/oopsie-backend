"""Per-lab feedback text from action_qa.json (run_action_qa.py).

Each lab gets an "Automated checks" block: one numbered item per flagged check, tagged with
the check's confidence and the action we ask for (``lab_feedback`` in action_qa_config.yaml),
the number of affected episodes and a few example episodes with what tripped the check.
Checks whose action is ``exclude`` never appear.

The block can stand alone (``lab_feedback_automated.txt``) or be appended to each lab's
section of the hand-written visual-review drafts (``lab_feedback_combined.txt``).
"""

from __future__ import annotations

import math
import re
import textwrap
from collections import Counter

WIDTH = 92

# Draft section names that differ from the dataset repo name (OopsieData-Submissions/<name>).
LAB_NAME_ALIASES = {
    "Enactic_OpenArm": "Enatic_OpenArm",
}

CONFIDENCE_LABELS = {"high_confidence": "High confidence", "lower_confidence": "Lower confidence"}
ACTION_LABELS = {"fix_and_resubmit": "Fix and resubmit", "confirm_and_discuss": "Confirm and talk to us"}
ACTION_ORDER = ("fix_and_resubmit", "confirm_and_discuss")

INTRO = """\
We ran automated checks on the actions, robot states and videos of {n} of your
episodes. Each finding is labelled with how reliable the check is:
  - High confidence: the check is precise, so a flagged episode almost certainly has the issue.
  - Lower confidence: the check is a heuristic, so it may flag episodes that are fine or miss
    some that aren't.
and with a suggested next step:
  - Fix and resubmit: the issue can be corrected in the data; please fix it and resubmit the
    affected episodes.
  - Confirm and talk to us: please check whether the issue is real and let us know, and we'll
    discuss how to handle it (for example, when the data can't be re-recorded)."""

NO_FINDINGS = "The automated checks found no issues in the {n} episodes we checked."

# (category, test) -> what the finding means and what we ask, as shown to labs.
MESSAGES: dict[tuple[str, str], str] = {
    ("general", "duplicate_pairs"):
        "Two different action keys hold identical values, which usually means one was copied "
        "into the other by mistake. Please record each action key from its own source.",
    ("general", "length_mismatch"):
        "An action and the robot state it is observed through have different numbers of "
        "steps. Please make every action and state array one entry per recorded step.",
    ("general", "action_chunk_shape"):
        "An arm action is stored with an extra axis, (steps, chunk, dof), i.e. the policy's "
        "whole action chunk was saved at every step. Please save only the action executed at "
        "each step, shaped (steps, dof).",
    ("general", "constant_action"):
        "An arm action never changes over the whole episode, which points to a stuck policy "
        "or a placeholder written in place of the real commands. Please confirm what was "
        "recorded and whether the real commands are available.",
    ("general", "command_rate"):
        "Each new arm action is held for the same number of steps throughout the episode, so "
        "commands arrive more slowly than the declared control_freq. Please confirm the real "
        "command rate and whether control_freq is correct.",
    ("general", "required_state_keys"):
        "A robot state the action space needs is missing (gripper_position always; "
        "joint_position for joint actions; cartesian_position for cartesian actions). Please "
        "add the missing state.",
    ("general", "placeholder_metadata"):
        "The operator, policy, robot or gripper name is blank, a template placeholder, or (for "
        "operator_name) not a person. Please fill in the real names.",
    ("general", "action_copies_state"):
        "The arm actions equal the recorded robot state at the same step, which suggests the "
        "actions were rebuilt from the states instead of logged as commands. Please confirm "
        "how the actions were recorded.",
    ("joint_position", "delta_detection"):
        "Joint position actions look like step-to-step changes rather than absolute joint "
        "positions. Please record absolute joint targets.",
    ("joint_position", "tracking"):
        "The commanded joint positions don't match where the joints end up on the next step. "
        "Please confirm that actions and states are aligned and refer to the same joints.",
    ("joint_position", "joint_names"):
        "The action/state joint names in the robot profile are missing, don't match each "
        "other, or are in an unusual order. Please confirm the joint names and their order.",
    ("joint_position", "dof"):
        "The number of joint values doesn't match the number of joint names in the robot "
        "profile. Please make them match.",
    ("joint_position", "radians"):
        "Joint values are larger than any angle in radians, so they look like degrees or a "
        "normalized range. Please convert joint positions to radians.",
    ("joint_position", "step_jump"):
        "A joint moves unrealistically far between two consecutive steps. Please confirm "
        "whether these jumps are real or come from dropped or misaligned steps.",
    ("joint_position", "gripper_in_joints"):
        "A gripper value is included in the joint vector. Please record it under the gripper "
        "action/state keys and leave it out of the joint vector.",
    ("joint_velocity", "dof"):
        "The number of joint velocities doesn't match the number of joint names in the robot "
        "profile. Please make them match.",
    ("joint_velocity", "tracking"):
        "The joint velocities times the step time don't match how far the joints actually "
        "moved. Please confirm the velocity units (rad/s) and the control_freq.",
    ("joint_velocity", "looks_like_position"):
        "The joint velocity values follow the joint positions, so they look like positions "
        "saved under the velocity key. Please store them under the correct key.",
    ("joint_velocity", "normalized_range"):
        "Joint velocities peak at exactly +-1, which suggests normalized values rather than "
        "rad/s. Please confirm the units.",
    ("joint_velocity", "speed_limit"):
        "A joint is commanded faster than a real arm can move. Please confirm the velocity "
        "units and values.",
    ("cartesian_position", "biarm_ordering"):
        "The two arms' cartesian actions match the robot state better with left and right "
        "swapped. Please confirm the arm order.",
    ("cartesian_position", "delta_detection"):
        "Cartesian xyz actions look like step-to-step changes rather than absolute positions. "
        "Please record absolute end-effector poses.",
    ("cartesian_position", "pose_tracking"):
        "The commanded end-effector position or orientation doesn't match where the robot "
        "ends up. Please confirm the frame and rotation convention of the actions.",
    ("cartesian_position", "position_units"):
        "Cartesian positions are not in metres (they look like cm/mm or a far-away frame). "
        "Please convert positions to metres in the robot base frame.",
    ("cartesian_position", "quaternion_shape"):
        "The cartesian pose is not [x, y, z, qx, qy, qz, qw] per arm with a unit-length "
        "quaternion. Please convert poses to that format.",
    ("cartesian_position", "rotation_jump"):
        "The commanded orientation turns unrealistically fast between two steps. Please "
        "confirm whether these jumps are real.",
    ("cartesian_velocity", "shape"):
        "Cartesian velocity isn't 6 values per arm (3 linear + 3 angular). Please convert it "
        "to that format.",
    ("cartesian_velocity", "speed_limit"):
        "The end effector is commanded faster than a real arm can move. Please confirm the "
        "velocity units and values.",
    ("cartesian_velocity", "normalized_range"):
        "Cartesian velocities peak at exactly +-1, which suggests normalized values rather "
        "than m/s and rad/s. Please confirm the units.",
    ("cartesian_velocity", "tracking"):
        "The cartesian velocities times the step time don't match how the end effector "
        "actually moved and turned. Please confirm the units, frame and control_freq.",
    ("cartesian_velocity", "linear_angular_order"):
        "The linear and angular halves of the cartesian velocity look swapped. Please confirm "
        "the order (linear first, then angular).",
    ("cartesian_velocity", "looks_like_position"):
        "The cartesian velocity values follow the end-effector position, so they look like "
        "positions saved under the velocity key. Please store them under the correct key.",
    ("video", "resolution"):
        "The video resolution is low. If higher-resolution originals are available, please "
        "replace the videos; otherwise, confirm this is the original recording resolution.",
    ("video", "decoding"):
        "A camera video is missing, fails to decode, or decodes to a different number of "
        "frames than its header states. Please re-export or repair the affected videos.",
    ("video", "brightness"):
        "A camera video is very dark or very bright. Please confirm whether the scene is "
        "visible as intended.",
    ("video", "fps"):
        "The video's frame rate doesn't match the episode's control_freq, so it plays back at "
        "the wrong speed. Please confirm which of the two is correct.",
    ("video", "duration"):
        "The video is longer or shorter than the recorded trajectory, so frames and steps are "
        "out of step. Please confirm how video frames map to recorded steps.",
    ("video", "camera_duration_mismatch"):
        "The cameras of one episode have different durations, so at least one camera is out "
        "of step with the others. Please confirm how the cameras were synchronized.",
    ("video", "corrupted_frames"):
        "Single video frames are unlike the frames on either side (e.g. a grey, green or "
        "garbled frame). Please re-export the affected videos from the source recording.",
    ("video", "video_jump"):
        "The image changes suddenly while the robot state changes smoothly, which points to a "
        "cut, dropped frames, or video and state out of step. Please confirm whether frames "
        "were dropped and how the video is synchronized with the state.",
    ("video", "choppy"):
        "The video repeats frames while the robot moves, so the motion looks choppy. Please "
        "confirm the camera's real frame rate and whether frames were duplicated on export.",
    ("video", "frozen"):
        "The video stops changing for a while as the robot keeps moving. Please confirm "
        "whether the camera stream was interrupted.",
    ("video", "periodic_skips"):
        "The robot state jumps at a regular interval throughout the motion, which usually "
        "means recording paused (e.g. during each policy inference) while the robot kept "
        "moving. Please confirm whether recording stopped during inference.",
}


def _num(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _flatten(details: dict, limit: int = 4) -> str:
    parts = []
    for k, v in details.items():
        if k == "ok":  # internal pass/fail flag, always False on a finding
            continue
        if isinstance(v, (int, float, str, bool)):
            parts.append(f"{k}={_num(v)}")
        elif isinstance(v, dict):
            inner = ", ".join(f"{a}={_num(b)}" for a, b in v.items() if not isinstance(b, (list, dict)))
            if inner:
                parts.append(f"{k} {inner}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            parts.append(", ".join(f"{a}={_num(b)}" for a, b in v[0].items() if not isinstance(b, (list, dict))))
        elif isinstance(v, list):
            parts.append(f"{k}={', '.join(map(str, v[:4]))}")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def _len(m: float) -> str:
    """A distance in metres, in cm below 1 m."""
    return f"{m * 100:.3g} cm" if abs(m) < 1 else f"{m:.3g} m"


def _hz(f: dict) -> str:
    return f" at {_num(f['control_freq'])} Hz" if f.get("control_freq") else ""


def _times(n: int, first_step) -> str:
    """'once, at step 146' / '8 times, first at step 216'."""
    return f"once, at step {first_step}" if n == 1 else f"{n} times, first at step {first_step}"


def _worst(items: list[dict], key: str) -> tuple[dict, int]:
    """The item with the largest ``key`` and how many others there are."""
    worst = max(items, key=lambda d: d.get(key) or 0)
    return worst, len(items) - 1


def _more(n: int, what: str) -> str:
    return f" (+{n} more {what}{'s' if n > 1 else ''})" if n else ""


def _arm(d: dict, items: list[dict]) -> str:
    """'arm 2 ' when the finding covers two arms, else ''."""
    return f"arm {d['arm']} " if len(items) > 1 or d.get("arm", 1) != 1 else ""


_JOINT_NAME_REASONS = {
    "missing_action_names": "no action joint names in the robot profile",
    "missing_state_names": "no state joint names in the robot profile",
    "action_names_not_in_state": "some action joints aren't in the state joint names",
    "reordered_vs_state": "actions and states list the joints in different orders",
    "biarm_sides_not_named": "joint names don't say which arm each joint is on",
}
_JOINT_NAME_PROBLEMS = {
    "blank_names": "include blank entries",
    "duplicate_names": "include duplicates",
    "numbering_out_of_order": "are numbered out of order",
    "chain_out_of_order": "are not in base-to-wrist order",
}

_TRACKING_REASONS = {
    "copies_state_motion": "exactly reproduces the recorded motion, so it was likely computed from the states",
    "too_large": "is {ratio:.1f}x the real motion",
    "too_small": "is only {ratio:.2f}x the real motion",
    "doesnt_match_state_motion": "barely follows the real motion (correlation {correlation:.2f})",
}
_SUSPECTED = {"deg_per_s": "looks like deg/s", "per_step_delta": "looks like per-step changes, not a rate"}


def _velocity_tracking(part: dict, reasons: list[str], prefix: str = "") -> str:
    if part.get("ratio") == 0:  # all-zero commands give no ratio or correlation to quote
        return f"{prefix}commanded velocity is zero while the robot moves"
    if not math.isfinite(part.get("correlation", 0)):
        reasons = [r for r in reasons if r != "doesnt_match_state_motion"]
    said = [_TRACKING_REASONS[r].format(**part) for r in reasons if r in _TRACKING_REASONS]
    if not said:
        return f"{prefix}commanded motion doesn't match the real motion"
    text = f"{prefix}commanded motion " + " and ".join(said)
    return text + (f" ({_SUSPECTED[part['suspected']]})" if part.get("suspected") in _SUSPECTED else "")


def _detail_pose_tracking(f: dict) -> str:
    parts = []
    for flag, arms_key, err_key, limit_key, fmt in (
        ("position_flagged", "position_arms", "median_error_m", "position_limit_m", _len),
        ("orientation_flagged", "orientation_arms", "median_error_deg", "orientation_limit_deg", lambda v: f"{v:.3g}°"),
    ):
        if f.get(flag) and f.get(arms_key):
            worst, _ = _worst(f[arms_key], err_key)
            what = "position" if arms_key == "position_arms" else "orientation"
            parts.append(f"{_arm(worst, f[arms_key])}{what} off by {fmt(worst[err_key])} at the median (limit {fmt(f[limit_key])})")
    return "; ".join(parts)


def _detail_position_units(f: dict) -> str:
    unit = lambda side: f", looks like {f[side + '_suspected_unit']}" if f.get(side + "_suspected_unit") else ""
    parts = []
    for r in f.get("reasons", []):
        if r == "action_out_of_range":
            parts.append(f"action positions reach {_num(f['action_max_abs_m'])}, too large for metres{unit('action')}")
        elif r == "state_out_of_range":
            parts.append(f"state positions reach {_num(f['state_max_abs_m'])}, too large for metres{unit('state')}")
        elif r == "action_state_scale_mismatch":
            ratio = f.get("action_state_ratio", math.inf)
            # An infinite ratio means the state never leaves zero (reports from before that
            # case stopped being flagged).
            parts.append(
                f"action positions are {ratio:.0f}x larger than the state's" if math.isfinite(ratio)
                else "the cartesian_position state is always zero"
            )
    return "; ".join(parts)


def _detail_quaternion_shape(f: dict) -> str:
    if f.get("reason") == "wrong_dim":
        return f"{f['actual_dim']} values per step instead of {f['expected_dim']}"
    if f.get("reason") == "bad_quaternion_norm":
        arms = f.get("bad_arms") or [{}]
        lo, hi = min(a.get("min_norm", 0) for a in arms), max(a.get("max_norm", 0) for a in arms)
        return f"quaternion length between {lo:.3g} and {hi:.3g} instead of 1"
    if f.get("reason") == "suspected_scalar_first":
        return "quaternion looks scalar-first (qw first instead of last)"
    return ""


def _detail_joint_names(f: dict) -> str:
    said = []
    for r in f.get("reasons", []):
        side, _, problem = r.partition("_")
        if r in _JOINT_NAME_REASONS:
            said.append(_JOINT_NAME_REASONS[r])
        elif side in ("action", "state") and problem in _JOINT_NAME_PROBLEMS:
            said.append(f"{side} joint names {_JOINT_NAME_PROBLEMS[problem]}")
    if f.get("not_in_state"):
        said.append("not in state: " + ", ".join(f["not_in_state"][:4]))
    return "; ".join(said)


def _detail_dof(f: dict) -> str:
    if "width" in f:  # joint_velocity: {width, names}
        return f"{f['width']} values but {f['names']} joint names"
    parts = [f"{side}s have {d['width']} values but {d['names']} joint names" for side, d in f.items() if side in ("action", "state")]
    if "action_vs_state" in f:
        d = f["action_vs_state"]
        parts.append(f"actions have {d['action_width']} values but states {d['state_width']}")
    return "; ".join(parts)


def _detail_radians(f: dict) -> str:
    parts = []
    for side in ("action", "state"):
        if side in f:
            unit = {"degrees": "degrees", "degrees_or_normalized_100": "degrees or a -100..100 range"}.get(f[side].get("suspected_unit"))
            parts.append(f"{side} values up to {_num(f[side]['max_abs'])}" + (f", looks like {unit}" if unit else ""))
    return "; ".join(parts) + f" (radians stay below {_num(f['limit_rad'])})"


def _detail_step_jump(f: dict) -> str:
    worst, more = _worst(f["joints"], "max_step_rad")
    return (
        f"{worst['joint']} moves {worst['max_step_rad']:.3g} rad in one step (limit {f['limit_step_rad']:.3g} rad"
        f"{_hz(f)}), {_times(worst['count'], worst['first_step'])}{_more(more, 'joint')}"
    )


def _detail_rotation_jump(f: dict) -> str:
    worst, _ = _worst(f["arms"], "max_step_deg")
    return (
        f"{_arm(worst, f['arms'])}turns {worst['max_step_deg']:.3g}° in one step (limit {f['limit_step_deg']:.3g}°"
        f"{_hz(f)}), {_times(worst['count'], worst['first_step'])}"
    )


def _detail_joint_tracking(f: dict) -> str:
    worst, more = _worst(f["worst_joints"], "median_error_rad")
    return f"{worst['joint']} off by {worst['median_error_rad']:.3g} rad at the median, limit {_num(f['limit_rad'])} rad{_more(more, 'joint')}"


def _detail_joint_speed(f: dict) -> str:
    worst, more = _worst(f["joints"], "max_speed")
    return f"{worst['joint']} commanded at {worst['max_speed']:.3g} rad/s (limit {_num(f['limit_rad_s'])} rad/s){_more(more, 'joint')}"


def _detail_cartesian_tracking(f: dict) -> str:
    parts = []
    for part in ("linear", "angular"):
        reasons = [r.removeprefix(part + "_") for r in f.get("reasons", []) if r.startswith(part + "_")]
        if reasons and part in f:
            parts.append(_velocity_tracking(f[part], reasons, f"{part}: "))
    frame = {"base": "robot base frame", "ee": "end-effector frame"}.get(f.get("frame"))
    if all("velocity is zero" in p for p in parts):
        frame = None  # no motion commanded, so no frame fits better than the other
    return "; ".join(parts) + (f" (best fit in the {frame})" if frame else "")


# (category, test) -> readable one-line detail. Unknown checks fall back to _flatten.
_DETAILS = {
    ("general", "duplicate_pairs"): lambda f: f"{f['key_a']} and {f['key_b']} are identical",
    ("general", "command_rate"): lambda f: (
        f"{f['key']} changes only every {f['period']} steps"
        + (f" (~{f['effective_hz']:g} Hz" if f.get("effective_hz") else " (")
        + f", control_freq {_num(f.get('control_freq'))} Hz)"
    ),
    ("general", "constant_action"): lambda f: (
        f"{f['key']} is the same for all {f['T']} steps"
        + (f" while {f['state_key']} moves {_num(f['state_range'])}" if f.get("state_range") else "")
    ),
    ("general", "placeholder_metadata"): lambda f: (
        f"{f['key']} is "
        + ("blank" if f.get("reason") == "blank" else f"'{f.get('value')}'")
        + {"placeholder": " (a placeholder)", "template": " (an unfilled template)", "not_a_person": " (not a person's name)"}.get(f.get("reason"), "")
    ),
    ("general", "required_state_keys"): lambda f: f"no {f['key']} state (needed by {', '.join(f.get('needed_by', []))})",
    ("general", "action_copies_state"): lambda f: (
        f"{f['key']} equals the "
        + ("next step's" if f.get("lag") == 1 else "same step's")
        + f" state in {f['matching_fraction']:.0%} of steps"
    ),
    ("general", "length_mismatch"): lambda f: f"{f['key']} has {f['action_T']} steps, {f['state_key']} state has {f['state_T']}",
    ("general", "action_chunk_shape"): lambda f: f"{f['key']} shaped {tuple(f['shape'])}",
    ("joint_position", "joint_names"): _detail_joint_names,
    ("joint_position", "dof"): _detail_dof,
    ("joint_position", "radians"): _detail_radians,
    ("joint_position", "step_jump"): _detail_step_jump,
    ("joint_position", "gripper_in_joints"): lambda f: "gripper joints in the joint vector: " + ", ".join(f["joints"]),
    ("joint_position", "tracking"): _detail_joint_tracking,
    ("joint_position", "delta_detection"): lambda f: f"fits step-to-step changes {1 / f['ratio']:.0f}x better than absolute positions",
    ("joint_velocity", "dof"): _detail_dof,
    ("joint_velocity", "speed_limit"): _detail_joint_speed,
    ("joint_velocity", "normalized_range"): lambda f: f"velocities peak at exactly ±{f['max_abs']:g}",
    ("joint_velocity", "tracking"): lambda f: _velocity_tracking(f, f.get("reasons", [])),
    ("joint_velocity", "looks_like_position"): lambda f: f"velocities sit within {f['position_match']:.0%} of the joint positions",
    ("joint_velocity", "smoothness"): lambda f: f"{f['sign_flip_rate']:.0%} of steps reverse direction",
    ("cartesian_position", "pose_tracking"): _detail_pose_tracking,
    ("cartesian_position", "position_units"): _detail_position_units,
    ("cartesian_position", "quaternion_shape"): _detail_quaternion_shape,
    ("cartesian_position", "rotation_jump"): _detail_rotation_jump,
    ("cartesian_position", "delta_detection"): lambda f: f"xyz fits step-to-step changes {1 / f['ratio']:.0f}x better than absolute positions",
    ("cartesian_position", "biarm_ordering"): lambda f: (
        f"positions off by {_len(f['same_arm_rms_m'])} as given, {_len(f['swapped_arm_rms_m'])} with the arms swapped"
    ),
    ("cartesian_velocity", "shape"): lambda f: f"{f['actual_dim']} values per step instead of {f['expected_dim']}",
    ("cartesian_velocity", "speed_limit"): lambda f: (
        f"up to {f['max_linear_m_s']:.3g} m/s linear and {f['max_angular_rad_s']:.3g} rad/s angular"
    ),
    ("cartesian_velocity", "normalized_range"): lambda f: f"velocities peak at exactly ±{f['max_abs']:g}",
    ("cartesian_velocity", "tracking"): _detail_cartesian_tracking,
    ("cartesian_velocity", "linear_angular_order"): lambda f: (
        f"agreement with the real motion {f['correlation_as_given']:.2f} as given vs {f['correlation_swapped']:.2f} with the halves swapped"
    ),
    ("cartesian_velocity", "looks_like_position"): lambda f: f"linear velocities sit within {f['position_match']:.0%} of the xyz positions",
    ("cartesian_velocity", "smoothness"): lambda f: f"{f['sign_flip_rate']:.0%} of steps reverse direction",
}


def finding_detail(f: dict) -> str:
    """One short, readable line on what tripped the check for this finding."""
    key = f.get("key", "")
    if "issue" in f:  # video checks already describe the problem in words
        issue = re.sub(r", first: .*", "", f["issue"])  # drop the raw decoder log line
        return issue if key in ("", "all cameras") else f"{key}: {issue}"
    formatter = _DETAILS.get((f["category"], f["test"]))
    if formatter:
        try:
            text = formatter(f)
            if text:
                return text
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass  # unexpected fields: fall back to the raw values below
    rest = {k: v for k, v in f.items() if k not in ("category", "test", "key")}
    flat = _flatten(rest)
    return f"{key}: {flat}" if flat else key


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, WIDTH, initial_indent="", subsequent_indent=indent)


def lab_name(dataset_repo: str) -> str | None:
    """Lab name for a submissions repo, or None for anything else (e.g. local qa_out bundles)."""
    org, _, name = dataset_repo.partition("/")
    return name if org == "OopsieData-Submissions" and name else None


def automated_block(repo: str, report: dict, lab_feedback: dict, n_examples: int = 3) -> str:
    """The "Automated checks" text for one dataset repo."""
    stats = report["per_dataset"][repo]
    n_checked = stats["episodes_checked"]
    # Episode IDs aren't unique within a repo (e.g. per-task numbering), so keep the path too.
    examples: dict[tuple[str, str], list[tuple[str, str, list[dict]]]] = {}
    repo_eps = [ep for ep in report["flagged_episodes"] if ep["dataset_repo"] == repo]
    id_counts = Counter(str(ep.get("episode_id") or "") for ep in repo_eps)
    for ep in repo_eps:
        by_test: dict[tuple[str, str], list[dict]] = {}
        for f in ep["findings"]:
            by_test.setdefault((f["category"], f["test"]), []).append(f)
        for ct, fs in by_test.items():
            examples.setdefault(ct, []).append((str(ep.get("episode_id") or ""), str(ep.get("repo_path") or ""), fs))

    items = []
    for (cat, test), eps in examples.items():
        fb = lab_feedback.get(f"{cat}.{test}", {"confidence": "lower_confidence", "action": "confirm_and_discuss"})
        if fb["action"] == "exclude":
            continue
        test_stats = stats["tests"].get(cat, {}).get(test, {})
        flagged = test_stats.get("flagged_episodes") or len(eps)
        checked = test_stats.get("checked_episodes") or n_checked
        items.append((ACTION_ORDER.index(fb["action"]), -flagged / max(checked, 1), cat, test, fb, flagged, checked, eps))
    items.sort(key=lambda t: t[:4])

    lines = ["Automated checks", "----------------", ""]
    if not items:
        lines.append(_wrap(NO_FINDINGS.format(n=n_checked), ""))
        return "\n".join(lines)
    lines += [INTRO.format(n=n_checked), ""]
    for i, (_, _, cat, test, fb, flagged, checked, eps) in enumerate(items, 1):
        tag = f"[{CONFIDENCE_LABELS[fb['confidence']]} · {ACTION_LABELS[fb['action']]}]"
        message = MESSAGES.get((cat, test), f"The {cat}.{test} check flagged these episodes. Please review them.")
        lines.append(_wrap(f"{i}. {tag} {message} Affected: {flagged} of {checked} checked episodes."))
        lines.append("   Example episode IDs:")
        for episode_id, path, fs in sorted(eps, key=lambda e: e[:2])[:n_examples]:
            if id_counts[episode_id] > 1 and path:  # ambiguous ID: name the file instead
                episode_id = path.removesuffix(".h5")
            detail = "; ".join(dict.fromkeys(finding_detail(f) for f in fs))
            lines.append(textwrap.fill(f"- {episode_id}: {detail}", WIDTH, initial_indent="     ", subsequent_indent="       "))
        lines.append("")
    return "\n".join(lines).rstrip()


def automated_report(report: dict, lab_feedback: dict, n_examples: int = 3) -> dict[str, str]:
    """Lab name -> automated block, for every submissions repo in the report."""
    out = {}
    for repo in sorted(report["per_dataset"], key=str.lower):
        name = lab_name(repo)
        if name:
            out[name] = automated_block(repo, report, lab_feedback, n_examples)
    return out


_HEADER_RULE = re.compile(r"^(=+|-+)\s*$")


def parse_drafts(text: str) -> list[tuple[str, str]]:
    """Split the visual-review drafts into ``(lab name, section text)`` in file order.
    A section starts at a non-empty line underlined with ``===`` (or ``---``)."""
    lines = text.splitlines()
    starts = [
        i for i in range(len(lines) - 1)
        if lines[i].strip() and not lines[i].startswith(" ") and _HEADER_RULE.match(lines[i + 1])
        and not _HEADER_RULE.match(lines[i])
    ]
    sections = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        sections.append((lines[i].strip(), "\n".join(lines[i:end]).rstrip()))
    return sections


def _header(name: str) -> str:
    return f"{name}\n{'=' * len(name)}"


def render_automated(blocks: dict[str, str]) -> str:
    return "\n\n\n".join(f"{_header(name)}\n\n{block}" for name, block in blocks.items()) + "\n"


def render_combined(drafts_text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Each draft section followed by its lab's automated block, then labs that only have
    automated results. Returns the text and the draft sections no dataset matched."""
    sections, used, unmatched = [], set(), []
    for name, text in parse_drafts(drafts_text):
        lab = LAB_NAME_ALIASES.get(name, name)
        if lab in blocks and lab not in used:  # a lab's block goes under its first section only
            used.add(lab)
            sections.append(f"{text}\n\n\n{blocks[lab]}")
        else:
            unmatched.append(name)
            sections.append(text)
    for lab, block in blocks.items():
        if lab not in used:
            sections.append(f"{_header(lab)}\n\n{block}")
    return "\n\n\n".join(sections) + "\n", unmatched
