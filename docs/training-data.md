# Training data

Training data is materialized from the episode index by recipes. Each recipe is a YAML file in `configs/training/`. It names which episodes to use, how to unify them, and which output formats to write. The nightly pipeline never materializes anything; every recipe is run on request.

See the [training package README](../oopsie_backend/training/README.md) for a file map and entry points.

```bash
uv run oopsie pipeline run training --set training_recipe=configs/training/droid_joint_velocity.yaml
uv run oopsie training materialize --recipe configs/training/eef_quat.yaml \
    --data-dir <mirror> --index episode_index.parquet --out <training dir>
sbatch slurm/pipeline.slurm training --set training_recipe=configs/training/all_native.yaml   # LS6
```

The pipeline target first brings the index, QA, critical flags, eval split and annotation columns up to date, then materializes the recipe. `oopsie training materialize` reads the index as it is.

## Steps

1. **Select** episodes from the index. These are always left out:
   - episodes not on disk, or unreadable;
   - duplicate copies;
   - critical episodes.

   The recipe's `selection` then filters the rest. `selection.splits` picks the eval split: `train` (the default) takes the episodes the eval split does not hold out, and `id`, `od_task_id_scene`, `od_task_od_scene` or `excluded` take the held-out ones, so a recipe can materialize the eval set.
2. **Compute** each robot's gripper range from the mirror.
3. **Unify** each episode into `episodes/<episode_key>.h5`, with one video per camera role in `episodes/<episode_key>/`. An episode that cannot be unified under the recipe is skipped, and the manifest records why. With `node_stage_dir` set, the episodes and their videos are copied there 1,000 at a time and each chunk's copies are removed once it is unified. In a multi-node job, the gripper ranges and the unification are split across the nodes.
4. **Write** `manifest.parquet`, `layout.json`, `recipe.yaml` and every format in `formats`.
5. **Record** the recipe in the index's `training_sets` column for every included episode.

The dataset is built in `<training_work_dir>/<recipe name>/`. If `training_dir` differs, the finished dataset is copied to `<training_dir>/<recipe name>/`, replacing the earlier copy. The unified episodes are copied only for the `hdf5` format. On LS6, the work directory is on `$SCRATCH` and the training directory on corral.

A unified episode is reused when all of these are unchanged since it was written:
- its source file;
- the recipe's unification settings (`rate_hz`, `upsample`, `actions`, `state`, `cameras`);
- the unification code;
- its robot's gripper range.

A change of `selection` or `formats` therefore only rewrites the outputs. The labels of the episodes live in the manifest, so new annotations do not require unifying again either.

## Recipes

```yaml
name: droid_joint_velocity         # snake_case; names the output folder and the RLDS dataset
formats: [rlds]                    # any of hdf5, lerobot, rlds
selection:
  repo_kinds: [contributor, external]
  splits: [train]                  # train, id, od_task_id_scene, od_task_od_scene, excluded
  outcome: all                     # all, success, failure, or annotated (any annotation)
  datasets: null                   # lists of dataset ids, lab ids or robot names; null means all
  exclude_datasets: []
  labs: null
  exclude_labs: []
  robots: [franka_panda, franka_research_3]
  biarm: false                     # the index's embodiment flags: true, false, or null for both
  binary_gripper: true
  dexterous_hand: null
  min_steps: 2
  min_seconds: 5                   # shortest episode in seconds; null keeps every length
  max_episodes: null               # stable pseudo-random subset, ranked by episode key and seed
  seed: 0
rate_hz: 15                        # target rate (default 15, as in DROID); null keeps every episode's own rate
upsample: interpolate              # episodes recorded below rate_hz: interpolate, hold, or native (keep their own rate)
actions:
  mode: target                     # target or native
  keys: [joint_velocity, gripper_position]
  orientation: quat                # quat, matrix, rot6d, rotvec or euler_<order>
  gripper: normalized              # normalized or raw
  from_states: never               # generate actions from the states: never, missing, or always
  pad_to: {}                       # minimum width per key, e.g. {joint_position: 7}
state:
  keys: [joint_position, gripper_position]
  pad_to: {}
cameras:
  image_size: [180, 320]           # height, width, as in DROID; null keeps the source resolution (hdf5 only)
  roles:                           # role: glob patterns over unified camera names, tried in order
    wrist_image_left: [wrist]
    exterior_image_1_left: [exterior_side, exterior_front, exterior_back, exterior_top]
    exterior_image_2_left: [exterior_side_2, exterior_front, exterior_side, exterior_back, exterior_top]
  optional: []                     # roles left blank when no camera matches, whatever `missing` says
  fallback:                        # a role that matched nothing takes another role's camera
    exterior_image_2_left: exterior_image_1_left
  missing: skip                    # skip the episode, or blank: black frames
  crf: 23                          # H.264 quality of the unified videos
```

An episode's boolean `success` comes from the latest annotation with a usable success score or outcome. Scores of at least 0.5 count as success. Dated annotations take priority, and ties use the first annotation in index order. `failure_categories` is the union of the annotations' categories. The manifest stores null when no annotation has a usable outcome.

The repo holds these example recipes:

| Recipe | Content | Formats |
|---|---|---|
| `droid_joint_velocity` | single-arm Franka, joint velocity plus gripper position, the DROID camera names `oopsie-rl` reads, 15 Hz | RLDS |
| `eef_quat` | single-arm, end-effector pose (xyz plus quaternion) plus gripper, 10 Hz, 224×224 | LeRobot |
| `all_native` | every usable episode with its own keys, masked, 10 Hz, blank wrist video where none exists | HDF5, LeRobot |
| `sample_images_rlds` | two wrist slots and one exterior slot with a head/body fallback at 10 Hz, with padding and a camera mask | RLDS |

### Image-only RLDS

In target mode, `actions.keys` and `state.keys` each select what to export.
The `sample_images_rlds` recipe leaves both unset, so the build reads camera
paths from HDF5 metadata and timing from the index's `num_steps` and `control_freq`.
Each step has RGB images, a camera mask, the reward, the discount, episode
flags and the language instruction. The export omits robot states and actions.

```bash
release=data/releases/2026-11-08
cp -n "$release/episode_index_sample.parquet" "$release/episode_index_images.parquet"
uv run oopsie index camera-names --index "$release/episode_index_images.parquet"
uv run oopsie training materialize \
    --recipe configs/training/sample_images_rlds.yaml \
    --data-dir "$release/sample_data" \
    --index "$release/episode_index_images.parquet" \
    --out "$release/training"
```

The camera mask follows the slot order `wrist_1`, `wrist_2`, `exterior_1`.
Each slot is a 180×320 RGB image. The role patterns use the index's
`unified_cameras` names. The exterior slot prefers front, side, top, then
back views. A head or body view fills the slot when no fixed exterior view
matches. Missing or unreadable views get black frames with a false mask entry.
An episode needs at least one readable assigned view. The manifest's `camera_sources` column
maps output roles back to source camera names.

`cameras.roles` controls the number of images per step. `rate_hz` controls
temporal sampling, and episode lengths remain variable.

Preview the assignment for a representative episode from each camera setup.

```bash
uv run oopsie index camera-sheet \
    --recipe configs/training/sample_images_rlds.yaml \
    --data-dir "$release/sample_data" \
    --index "$release/episode_index_images.parquet" \
    --out "$release/analysis/sample_images_camera_sheet.html"
```

Use TensorFlow Datasets to read a step and check its camera mask.

```python
import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(
    "data/releases/2026-11-08/training/sample_images_rlds/rlds/sample_images_rlds/1.0.0"
)
episode = next(iter(builder.as_dataset(split="train")))
observation = next(iter(episode["steps"]))["observation"]
image = observation["exterior_1"]
image_is_present = observation["camera_mask"][2]
```

## Unification

### Actions and states

In **target** mode, `actions.keys` and `state.keys` are the keys every episode must provide. An omitted or empty list leaves those fields out of RLDS. LeRobot needs both lists. A key that an episode lacks is derived where that needs no kinematics:

| Key | Derived from |
|---|---|
| `joint_position` | joint state + `joint_velocity` / rate |
| `joint_velocity` | (`joint_position` action − joint state) × rate |
| `cartesian_position` | pose state advanced by `cartesian_velocity` / rate |
| `cartesian_velocity` | [(position action − state position) × rate, rotation vector of R_action R_state⁻¹ × rate] |
| `gripper_position` | `gripper_binary` |
| `gripper_binary` | normalized `gripper_position` > 0.5 |

Derivations use the source rate, before resampling. An episode whose keys cannot be provided is skipped; the manifest's `reason` says which key was missing and what the episode has.

With `actions.from_states`, actions are generated from the states: every action key with `always`, or only those an episode cannot provide otherwise with `missing`. They are computed on the resampled steps, from each step's state and the next step's state:

| Key | Generated from |
|---|---|
| `joint_position`, `cartesian_position`, `gripper_position` | the next step's state |
| `joint_velocity` | (next joint state − joint state) × rate |
| `cartesian_velocity` | [(next position − position) × rate, rotation vector of R_next R⁻¹ × rate] |
| `gripper_binary` | next gripper state > 0.5 |

The last step has no next state and is dropped. The attr `actions_from_states` and the manifest column of the same name list the generated keys.

In **native** mode, every episode keeps the action keys of its `action_space` and its `joint_position`, `cartesian_position` and `gripper_position` states. Native `cartesian_velocity` is passed through unchanged.

In both modes:
- Cartesian poses are written in `actions.orientation`, per arm. A pose block of width 7 is read as a quaternion pose, as the Oopsie recorder writes it. Other widths follow the profile's orientation representation.
- Gripper positions are scaled to [0, 1] with the minimum and maximum over all selected episodes of the same robot and gripper (`gripper: normalized`).
- A gripper key must have one value per arm, so episodes with `dexterous_hand` set are skipped. Recipes can leave them out up front with `selection.dexterous_hand: false` or `binary_gripper: true`.

Not unified: joint units (some labs record degrees), the gripper's open and closed direction, and robot kinematics.

### Rate

`rate_hz` is the target rate and defaults to 15 Hz, the DROID rate. Actions and states are converted at the episode's own rate, then resampled; velocities keep their units, so they stay valid at the new rate.

- **Downsampling** takes, for each output step, the latest source step at or before it.
- **Upsampling** depends on `upsample`:
  - `interpolate` (default): `joint_position` and `gripper_position` are interpolated linearly, and `cartesian_position` linearly in position and by slerp in rotation. Velocities, `gripper_binary` and base keys are held, since a velocity command applies to its whole source interval.
  - `hold`: every value is held.
  - `native`: the episode keeps its own rate.

Videos get one frame per output step, the frame of the latest source step at or before it; frames are never interpolated. Every unified episode has a per-step `filled` flag that marks the steps that add no new source step, so training can mask or down-weight them. The manifest's `filled_steps` counts them.

### Layout and masks

`layout.json` fixes the key order and the width of every key, which is the largest width any included episode has, or `pad_to`. The `action` and `state` vectors concatenate the keys in that order, padded with zeros. `action_mask` and `state_mask` mark the entries an episode actually has, so a 6-DoF arm and a 7-DoF arm, or a single arm and a bimanual setup, share one dataset.

### Cameras

Camera roles match the [unified camera names](camera-names.md) of the index's `unified_cameras` column, not the names the episodes record. Each role takes the first camera whose unified name matches one of its patterns, in order, and a camera fills at most one role. A camera that is left out or marked `unknown` has no unified name and fills no role. An episode without any named camera, or one the camera names stage has not seen, is skipped. The unified episodes record the camera names they were written with, so a changed name rewrites them.

`cameras.preset: two_exterior_two_wrist` replaces `roles` and `optional` with up to two wrist and two scene cameras, all optional:

| Role | Optional | Unified names, in order |
|---|---|---|
| `wrist_1` | yes | `wrist`, `wrist_left` |
| `wrist_2` | yes | `wrist_right`, `wrist_2` |
| `exterior_1` | yes | `exterior_front`, `exterior_side`, `exterior_top`, `exterior_back`, their `_2` cameras, `head` |
| `exterior_2` | yes | `exterior_side`, `exterior_top`, `exterior_front`, `exterior_back`, their `_2` cameras, `head` |

The first scene camera is the most informative placement, a perspective view of gripper and objects before a top-down one; the second is a different placement where the episode has one, so the two show the scene from two sides. A single-arm robot fills `wrist_1` and, with a second wrist camera, `wrist_2`; a bimanual robot puts its left wrist in `wrist_1` and its right in `wrist_2`. Stereo images are never used, as they repeat their first image.

An episode in which no role has a camera is skipped. A blank role holds black frames. Every format marks which roles have a camera, in role order: `observation.camera_mask` per frame in LeRobot, `observation/camera_mask` per step in RLDS, and an empty `camera/<role>` attr in HDF5. A policy should mask a blank camera's input rather than read the black frames.

## Formats

**`hdf5`**: the unified episodes themselves, together with `layout.json` and `manifest.parquet`.

| Location | Content |
|---|---|
| `episodes/<episode_key>.h5`: `action/<key>`, `state/<key>` | `(T, D)` float32, unpadded |
| `episodes/<episode_key>.h5`: `filled` | `(T,)` bool |
| `episodes/<episode_key>.h5` attrs | `fps`, `source_fps`, `num_steps`, `actions_from_states`, `camera/<role>` (the video path, empty for a blank role) and `camera_sources` |
| `manifest.parquet` | every selected episode with `status` (`written`, `reused`, `skipped`, `failed`), `reason`, labels and widths |

**`lerobot`**: a [LeRobot](https://github.com/huggingface/lerobot) v2.1 dataset in `lerobot/`.
- Frames hold `observation.state`, `action`, their masks, the time and index columns, and one video feature `observation.images.<role>` per role.
- `meta/episodes.jsonl` also carries each episode's key, source, lab, robot, success and failure categories.
- Videos are hard links to the unified videos.
- Frames also hold `filled`.
- LeRobot needs a single frame rate, so `rate_hz: null` or `upsample: native` work only when every episode ends up at the same rate.

**`rlds`**: a TFDS dataset at `rlds/<recipe name>/1.0.0`.
- Each step holds:
  - `observation/<role>` (JPEG), `observation/state` and `observation/<state key>`;
  - `action`, `action_dict/<key>` and `is_filled`;
  - `reward` (1 on the last step of a successful episode), `discount`, `is_first`, `is_last`, `is_terminal` and `language_instruction`.
- `episode_metadata` holds the source, boolean `success`, failure categories and masks. Episodes with no usable outcome have `success` set to `false`.
- `file_path` starts with `success/` or `failure/`, from which `oopsie-rl` infers the outcome.
- Recipes that use the DROID camera names and keys produce the layout `oopsie-rl` reads.

To add a format, write `oopsie_backend/training/writers/<format>.py` with `write(episodes_dir, out, included, layout, recipe)`, and add it to `training/writers/registry.py` and `FORMATS` in `training/recipe.py`.
