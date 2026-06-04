# oopsiedata_format_v1 — Full Schema Reference

Use this skill whenever writing or debugging a converter that produces Oopsie HDF5 episode files. It describes the canonical `oopsiedata_format_v1` schema as enforced by `oopsie_tools`.

You follow these steps:

1. Create a new config robot profile. Double check with the user that this robot profile is correct and no unverified information has been entered. The profile should be placed in ./configs/
2. Create the conversion script in ./oopsie_backend/external_conversion/. Follow DRY and make sure to reuse utilities from ./oopsie_backend/external_conversion/conversion_utils.py. Make sure to read all relevant information before attempting to write the conversion script.
3. Wait for the user to run the conversion script with the full correct input path.

## Validation entry points (`oopsie_tools`)

```
oopsie_tools/utils/validation/episode_loader.py   # HDF5 → EpisodeData
oopsie_tools/utils/validation/episode_validator.py # semantic checks
oopsie_tools/utils/validation/validation_utils.py  # public API
oopsie_tools/utils/robot_profile/robot_profile.py  # RobotProfile
```

Run validation:
```bash
python scripts/validate_and_upload/validate.py --path /path/to/directory
```

---

## HDF5 File Structure

### Root attributes (all required)

| Attribute | Type | Notes |
|---|---|---|
| `schema` | str | Must be `"oopsiedata_format_v1"` |
| `episode_id` | str | Non-empty unique identifier |
| `language_instruction` | str | Non-empty task description |
| `lab_id` | str | Non-empty; not `"your_lab_id"` |
| `operator_name` | str | Non-empty |
| `robot_profile` | str | JSON-serialized `RobotProfile` (use `robot_profile_to_json`) |

### `/observations/` group

**`/observations/video_paths/`** — one string dataset per camera:
- Key = camera name (must match `profile.camera_names`)
- Value = relative path to MP4 from the HDF5 file's directory

**`/observations/robot_states/`** — one dataset per robot state key:
- Keys must include all `profile.robot_state_keys` (required: `joint_position`, `gripper_position`)
- Shape: `(T, D)` float64
- `joint_position`: `(T, n_joints)` — names from `robot_state_joint_names`
- `gripper_position`: `(T, 1)` or `(T, n_fingers)`
- `cartesian_position`: `(T, 6)` if included (xyz + euler angles)

### `/actions/` group

- Keys in `profile.action_space`: stored as `(T, D)` float64 arrays (non-empty)
- All other canonical action keys: stored as `h5py.Empty(dtype=np.float64)`

Canonical action keys (write Empty for any not in action_space):
```
cartesian_position, cartesian_velocity,
joint_position, joint_velocity,
base_position, base_velocity,
gripper_velocity, gripper_position, gripper_binary
```

### `/episode_annotations/` group (optional for writing, required for upload)

Structure: `episode_annotations/{annotator_name}/` — a group with HDF5 attributes:

| Attr | Type | Notes |
|---|---|---|
| `success` | float | Required; must be in [0.0, 1.0] |
| `failure_description` | str | Optional; if set, must also set `taxonomy` |
| `taxonomy` | str | JSON: `{"failure_category": [...], "severity": "..."}` |

Either all three of `(failure_category via taxonomy, failure_description, severity)` are filled, or all three are empty.

---

## RobotProfile fields

Defined in `oopsie_tools/utils/robot_profile/robot_profile.py`. Serialized as JSON into `f.attrs["robot_profile"]`.

**Required fields:**

| Field | Type | Example |
|---|---|---|
| `policy_name` | str | `"pi0_droid"` |
| `robot_name` | str | `"franka_research_3"` |
| `gripper_name` | str | `"robotiq_2f_85"` |
| `is_biarm` | bool | `false` |
| `uses_mobile_base` | bool | `false` |
| `control_freq` | int | `10` |
| `camera_names` | list[str] | `["left", "right", "wrist"]` |
| `robot_state_keys` | list[str] | Must include `"joint_position"` and `"gripper_position"` |
| `robot_state_joint_names` | list[str] | One entry per joint DOF in `joint_position` |
| `action_space` | list[str] | Exactly 1 arm key + 1 gripper key (see below) |

**`action_space` validity rules:**
- Exactly ≥1 arm key from: `{joint_position, joint_velocity, cartesian_position, cartesian_velocity}`
- Exactly ≥1 gripper key from: `{gripper_position, gripper_velocity, gripper_binary}`
- Optionally 1 base key from: `{base_velocity, base_position}`
- No other keys allowed

**Optional fields:**

| Field | Notes |
|---|---|
| `action_joint_names` | Required if `action_space` includes `joint_position` or `joint_velocity` |
| `robot_state_orientation_representation` | Required if `cartesian_position` in `robot_state_keys`. Options: `"euler_xyz"`, `"quat"`, `"rot6d"`, `"rotvec"`, `"matrix"` |
| `orientation_representation` | Required if action space includes `cartesian_position` or `cartesian_velocity` |
| `controller` | str — `"OSC"`, `"joint_position"`, `"joint_velocity"` |

---

## Validation constraints

| Check | Limit |
|---|---|
| Video min dimension | 180 px |
| Video max dimension | 1280 px |
| Episode duration (trajectory_length / control_freq) | 1 – 300 seconds |
| Frame count vs trajectory_length tolerance | max(5, 10% of T) |
| Video duration vs expected duration | ≤ 0.5 s |
| All obs/action arrays same length T | strict |

---

## Minimal write pattern (Python)

```python
import h5py, numpy as np
from oopsie_tools.utils.robot_profile.robot_profile import RobotProfile, robot_profile_to_json

SCHEMA = "oopsiedata_format_v1"
ALL_ACTION_KEYS = [
    "cartesian_position", "cartesian_velocity",
    "joint_position", "joint_velocity",
    "base_position", "base_velocity",
    "gripper_velocity", "gripper_position", "gripper_binary",
]

profile = RobotProfile(
    policy_name="my_policy",
    robot_name="franka_research_3",
    gripper_name="robotiq_2f_85",
    is_biarm=False,
    uses_mobile_base=False,
    control_freq=10,
    camera_names=["left", "right", "wrist"],
    robot_state_keys=["joint_position", "gripper_position"],
    robot_state_joint_names=[f"joint_{i}" for i in range(1, 8)],
    action_space=["joint_velocity", "gripper_position"],
    action_joint_names=[f"joint_{i}" for i in range(1, 8)],
)

str_dtype = h5py.string_dtype(encoding="utf-8")

with h5py.File("episode.h5", "w") as f:
    f.attrs["schema"] = SCHEMA
    f.attrs["episode_id"] = "episode_000001"
    f.attrs["language_instruction"] = "Pick up the cup"
    f.attrs["lab_id"] = "UT Austin"
    f.attrs["operator_name"] = "Alice"
    f.attrs.create("robot_profile", robot_profile_to_json(profile), dtype=str_dtype)

    obs = f.create_group("observations")
    vp = obs.create_group("video_paths")
    for cam in ["left", "right", "wrist"]:
        vp.create_dataset(cam, data=f"videos/episode_000001_{cam}.mp4", dtype=str_dtype)

    rs = obs.create_group("robot_states")
    rs.create_dataset("joint_position", data=joint_pos_array)   # (T, 7)
    rs.create_dataset("gripper_position", data=gripper_pos_array)  # (T, 1)

    acts = f.create_group("actions")
    for key in ALL_ACTION_KEYS:
        if key == "joint_velocity":
            acts.create_dataset(key, data=joint_vel_array)   # (T, 7)
        elif key == "gripper_position":
            acts.create_dataset(key, data=gripper_act_array)  # (T, 1)
        else:
            acts.create_dataset(key, data=h5py.Empty(dtype=np.float64))

    # Annotations (required for upload, optional otherwise)
    ea = f.create_group("episode_annotations")
    ann = ea.create_group("my_annotator")
    ann.attrs["success"] = np.float64(1.0)
```
