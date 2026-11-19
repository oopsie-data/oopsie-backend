# Action checks

The action checks read the full `actions/*` and `observations/robot_states/*` arrays of an episode and flag values that contradict the Oopsie-Data format or the robot's own motion. Most problems they find come from export scripts: wrong units, positions stored as velocities, swapped arms, normalized and clipped values.

Each table below is one section of `configs/action_qa_config.yaml`. Parameter names are given with their defaults in parentheses, and the Default column says whether the check is enabled. How checks run and how their results are stored is described on the [QA](index.md) page.

## Conventions

- **Categories.** `general` runs on every episode with a declared `action_space`. Each arm category (`joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`) runs only if that key is declared in the robot profile's `action_space`.
- **Matching state.** Each action is compared with the state it is observed through: position actions with the same-named state, velocity actions with the corresponding position state (`joint_velocity` with `joint_position`, `cartesian_velocity` with `cartesian_position`, `gripper_*` with `gripper_position`).
- **Tracking alignment.** "Tracking" compares `action[t]` with `state[t+1]`, the state the command should lead to. Actions and states are truncated to the shorter length; a length difference is reported by `length_mismatch`.
- **Joint pairing.** Joint actions are paired with state joints by name when the profile's `action_joint_names` and `robot_state_joint_names` match the array widths and every action joint appears in the state; otherwise by column index when the widths agree. Without a pairing the state-based joint checks are skipped (`no_joint_pairing`).
- **Gripper columns.** Joints whose name contains `gripper` are reported by `joint_position.gripper_in_joints` and excluded from all other joint checks.
- **Per-step limits.** Speed limits in rad/s or deg/s are converted to a per-step limit by dividing by the profile's `control_freq`; without it those checks are skipped (`no_control_freq`).
- **Normalized range.** `normalized_range` flags values whose maximum absolute value is exactly 1.0 (to 1e-6): the signature of values normalized to [-1, 1] and clipped (DROID-style). Values normalized without clipping, or to another range, are not detected by it.
- **Radians.** `joint_position.step_jump` and `tracking` assume radians: they are skipped as `not_radians` when the action exceeds `radians.max_abs_rad`, whether or not the `radians` check itself is enabled.
- **Skips.** A check that cannot run on an episode records a skip with a reason instead of passing, e.g. `declared_but_empty` (the key is declared but the dataset is `h5py.Empty`), `no_matching_state`, `non_finite_values`, `too_short`, `too_little_motion`.

## `general`

| Check | Default | Flags when |
|---|---|---|
| `duplicate_pairs` | on | two declared action arrays of the same shape are equal to within `atol` (1e-9) / `rtol` (0) |
| `length_mismatch` | on | an action and its matching state have different numbers of timesteps |
| `action_chunk_shape` | on | an arm action has shape `(T, chunk, dof)`: each step holds a policy's whole action chunk instead of the executed action |
| `constant_action` | on | an arm action never changes over the episode to within `atol` (1e-9): a stuck policy or placeholder actions |
| `command_rate` | on | each new arm action is held for the same N ≥ 2 steps in at least `min_fraction` (0.8) of holds, so commands arrived at `control_freq / N`. Skipped with fewer than `min_holds` (10) changes |
| `required_state_keys` | on | a state the action space needs is missing: `gripper_position` always, `joint_position` or `cartesian_position` for joint or Cartesian actions |
| `placeholder_metadata` | on | the operator, policy, robot or gripper name is blank, a placeholder (`placeholder_values`, `operator_placeholder_values`) or an unfilled template (`template_prefixes`, e.g. `your_`) |
| `action_copies_state` | on | `joint_position` or `cartesian_position` action equals the same-named state, at the same step or the next, to within `atol` (1e-6) in at least `min_fraction` (0.9) of steps. Skipped when the state never moves (`state_static`) |
| `static_robot` | on | the arm barely moves: no non-gripper `joint_position` state spans `max_joint_range_rad` (0.05) and no `cartesian_position` xyz spans `max_position_range_m` (0.02 m). Spans run between the `quantile` (0.01) and 1 - `quantile` quantiles. A joint state beyond `max_abs_rad` (6.5) or an xyz beyond `max_abs_position_m` (5) is left out; with neither state usable the check is skipped (`no_arm_state`) |
| `gripper_binary_present` | on | the episode has no non-empty `gripper_binary` action. Reports the gripper keys declared and recorded instead |
| `gripper_range` | on | `gripper_binary` action falls outside [0, 1] by more than `atol` (0.05), e.g. -1/+1 encoding |
| `gripper_command_range` | on | `gripper_position` action exceeds the episode's `gripper_position` state range by more than `max_fraction` (0.25) of the state's scale (the larger of its span and its largest absolute value), with an allowed margin of at least `min_margin` (0.05) |
| `scale_consistency` | off | `gripper_position`: RMS(action) / RMS(state) is above `max_ratio` (5) or below `min_ratio` (0.2) |
| `tracking_residual` | off | `gripper_position`: RMS(action[t] - state[t+1]) / RMS(action) is above `max_normalized_rms` (0.5) |
| `delta_detection` | off | `gripper_position`: the action fits state[t+1] - state[t] much better than state[t+1], i.e. RMS(delta residual) / RMS(absolute residual) is below `ratio_threshold` (0.3). Skipped for a constant action |

## `joint_position`

| Check | Default | Flags when |
|---|---|---|
| `joint_names` | on | the profile's action or state joint names are missing, blank or duplicated; numbered joints are out of order (`joint3` before `joint2`); standard UR/ViperX/SO-100 joints are out of base-to-wrist order (e.g. an alphabetical ROS list); an action joint is missing from the state or in a different relative order; or a biarm robot's names don't say which arm each joint is on |
| `dof` | on | the action or state width differs from its number of joint names, or, without names, action and state widths differ |
| `gripper_in_joints` | on | a joint name contains `gripper` (the gripper belongs in its own action key) |
| `normalized_range` | on | non-gripper joint actions peak at exactly ±1 |
| `radians` | on | any action or state joint value exceeds `max_abs_rad` (6.5). Reports a suspected unit: `degrees_or_normalized_100` up to 100.5 (SO-100/101 arms store either), `degrees` up to 400 |
| `step_jump` | on | a joint moves more than `max_joint_speed_rad_s` (6.28) / `control_freq` between two steps. Skipped if the action exceeds `radians.max_abs_rad` |
| `delta_detection` | on | same test as `general.delta_detection`, over the paired joints |
| `tracking` | on | the median \|action[t] - state[t+1]\| of any paired joint exceeds `max_median_error_rad` (0.2). Skipped if the action exceeds `radians.max_abs_rad` |

## `joint_velocity`

The velocity is integrated over one control step (`v / control_freq`) and compared with the per-step change of the joint position state.

| Check | Default | Flags when |
|---|---|---|
| `dof` | on | the action width differs from the number of joint names |
| `speed_limit` | on | any joint's peak \|velocity\| exceeds `max_joint_speed_rad_s` (3.14) |
| `normalized_range` | on | velocities peak at exactly ±1 |
| `smoothness` | on | more than `max_sign_flip_rate` (0.3) of consecutive steps flip sign, counting only steps where both values exceed 10% of that joint's RMS |
| `looks_like_position` | on | RMS(v[t] - state[t+1]) / RMS(state[t+1]) is below `max_position_match` (0.2), i.e. the "velocity" is a position |
| `tracking` | on | the commanded per-step motion vs. the state's per-step motion (best of lag 0 and 1): RMS ratio below `min_ratio` (0.25) or above `max_ratio` (8), correlation below `min_correlation` (0.3), or an exact copy of the state's motion. Reports a suspected cause: `deg_per_s` for a ratio near 57, `per_step_delta` for a ratio near 1 / `control_freq`. Skipped when no joint's range exceeds `min_motion_rad` (0.2) |

## `cartesian_position`

Each arm's action must be `[x, y, z, qx, qy, qz, qw]` (scalar-last). The state may store orientation in another representation; it is converted with the profile's `robot_state_orientation_representation`, and a 4-wide state rotation is read as a quaternion.

| Check | Default | Flags when |
|---|---|---|
| `quaternion_shape` | on | the action width is not 7 (14 biarm), or a quaternion's norm differs from 1 by more than `norm_atol` (1e-3). All other checks below are skipped for a wrong width |
| `normalized_range` | on | action xyz peaks at exactly ±1. Quaternion components are excluded, since they reach ±1 legitimately |
| `position_units` | on | action or state xyz exceeds `max_abs_position_m` (3.0), or RMS(action xyz) / RMS(state xyz) exceeds `max_action_state_ratio` (10). Reports a suspected unit (`cm`/`mm`) from that ratio when the state looks like metres, else from which unit puts the values within a typical reach of `typical_reach_min_m`–`typical_reach_max_m` (0.2–1.5 m) |
| `quaternion_sign_flip` | off | consecutive action quaternions have a negative dot product (the same orientation written as -q) |
| `rotation_jump` | on | the action rotates more than `max_angular_speed_deg_s` (360) / `control_freq` between two steps |
| `quaternion_scalar_order` | off | heuristic: the first quaternion component looks more like a near-constant scalar than the last, suggesting `[qw, qx, qy, qz]`. Misfires on gripper-down poses |
| `delta_detection` | on | same test as `general.delta_detection`, on xyz |
| `biarm_ordering` | on | biarm only: the RMS error of action xyz vs. state xyz with the two arms swapped is below `max_swap_ratio` (0.5) times the error as given |
| `pose_tracking` | on | per arm, the median distance between action[t] and state[t+1] xyz exceeds `max_median_position_error_m` (0.10), or the median rotation angle between them exceeds `max_median_orientation_error_deg` (20). The orientation half is skipped when the state orientation can't be converted (`unknown_state_orientation`) |

## `cartesian_velocity`

Each arm's action must be 6 values, `[vx, vy, vz, wx, wy, wz]`. The `tracking` check compares against the `cartesian_position` state's per-step motion in both the base and the end-effector frame and reports the frame that fits better; the frame itself is not judged. Without a convertible state orientation only the linear part is checked.

| Check | Default | Flags when |
|---|---|---|
| `shape` | on | the action width is not 6 (12 biarm). All other checks are skipped for a wrong width |
| `speed_limit` | on | peak linear speed exceeds `max_linear_m_s` (2.0) or peak angular speed exceeds `max_angular_rad_s` (6.28) |
| `normalized_range` | on | velocities peak at exactly ±1 |
| `smoothness` | on | same test as `joint_velocity.smoothness` |
| `looks_like_position` | on | same test as `joint_velocity.looks_like_position`, on the linear part vs. the state's xyz |
| `tracking` | on | same test as `joint_velocity.tracking`, separately for the linear part (judged only if the end effector moves at least `min_translation_m` (0.05)) and the angular part (only if it turns at least `min_rotation_deg` (15)) |
| `linear_angular_order` | on | swapping the linear and angular halves raises the mean correlation with the state's motion by more than `min_improvement` (0.2). Requires both translation and rotation above the `tracking` minimums |
