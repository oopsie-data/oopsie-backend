# Training data

Training data is materialized from the episode index by recipes. Each recipe is a YAML file in `configs/training/`. It names which episodes to use, how to unify them, and which output formats to write. The nightly pipeline never materializes anything; every recipe is run on request.

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
   - critical episodes;
   - episodes held out by the eval split, including `excluded`.

   The recipe's `selection` then filters the rest.
2. **Compute** each robot's gripper range from the mirror.
3. **Unify** each episode into `episodes/<episode_key>.h5`, with one video per camera role in `episodes/<episode_key>/`. An episode that cannot be unified under the recipe is skipped, and the manifest records why. With `node_stage_dir` set, the episodes and their videos are copied there 1,000 at a time and each chunk's copies are removed once it is unified. In a multi-node job, the gripper ranges and the unification are split across the nodes.
4. **Write** `manifest.parquet`, `layout.json`, `recipe.yaml` and every format in `formats`.
5. **Record** the recipe in the index's `training_sets` column for every included episode.

The dataset is built in `<training_work_dir>/<recipe name>/`. If `training_dir` differs, the finished dataset is copied to `<training_dir>/<recipe name>/`, replacing the earlier copy. The unified episodes are copied only for the `hdf5` format. On LS6, the work directory is on `$SCRATCH` and the training directory on corral.

A unified episode is reused when all of these are unchanged since it was written:
- its source file;
- the recipe's unification settings (`rate_hz`, `actions`, `state`, `cameras`);
- the unification code;
- its robot's gripper range.

A change of `selection` or `formats` therefore only rewrites the outputs. The labels of the episodes live in the manifest, so new annotations do not require unifying again either.

## Recipes

```yaml
name: droid_joint_velocity         # snake_case; names the output folder and the RLDS dataset
formats: [rlds]                    # any of hdf5, lerobot, rlds
selection:
  repo_kinds: [contributor, external]
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
  max_episodes: null               # stable pseudo-random subset, ranked by episode key and seed
  seed: 0
rate_hz: 15                        # null keeps every episode's own rate
actions:
  mode: target                     # target or native
  keys: [joint_velocity, gripper_position]
  orientation: quat                # quat, matrix, rot6d, rotvec or euler_<order>
  gripper: normalized              # normalized or raw
  pad_to: {}                       # minimum width per key, e.g. {joint_position: 7}
state:
  keys: [joint_position, gripper_position]
  pad_to: {}
cameras:
  image_size: [180, 320]           # height, width
  roles:                           # role: glob patterns over camera names, tried in order
    wrist_image_left: ["*wrist*"]
    exterior_image_1_left: [left, "*exterior*", "*"]
    exterior_image_2_left: [right, "*exterior*", "*"]
  fallback:                        # a role that matched nothing takes another role's camera
    exterior_image_2_left: exterior_image_1_left
  missing: skip                    # skip the episode, or blank: black frames
  crf: 23                          # H.264 quality of the unified videos
```

An episode's `success` is the mean success of its annotations, falling back to the annotation outcome. Its `failure_categories` is the union of the annotations' categories. `outcome: success` keeps episodes with success ≥ 0.5.

The repo holds three example recipes:

| Recipe | Content | Formats |
|---|---|---|
| `droid_joint_velocity` | single-arm Franka, joint velocity plus gripper position, the DROID camera names `oopsie-rl` reads, 15 Hz | RLDS |
| `eef_quat` | single-arm, end-effector pose (xyz plus quaternion) plus gripper, 10 Hz, 224×224 | LeRobot |
| `all_native` | every usable episode with its own keys, masked, 10 Hz, blank wrist video where none exists | HDF5, LeRobot |

## Unification

### Actions and states

In **target** mode, `actions.keys` and `state.keys` are the keys every episode must provide. A key that an episode lacks is derived where that needs no kinematics:

| Key | Derived from |
|---|---|
| `joint_position` | joint state + `joint_velocity` / rate |
| `joint_velocity` | (`joint_position` action − joint state) × rate |
| `cartesian_position` | pose state advanced by `cartesian_velocity` / rate |
| `cartesian_velocity` | [(position action − state position) × rate, rotation vector of R_action R_state⁻¹ × rate] |
| `gripper_position` | `gripper_binary` |
| `gripper_binary` | normalized `gripper_position` > 0.5 |

Derivations use the source rate, before resampling. An episode whose keys cannot be provided is skipped; the manifest's `reason` says which key was missing and what the episode has.

In **native** mode, every episode keeps the action keys of its `action_space` and its `joint_position`, `cartesian_position` and `gripper_position` states. Native `cartesian_velocity` is passed through unchanged.

In both modes:
- Cartesian poses are written in `actions.orientation`, per arm. A pose block of width 7 is read as a quaternion pose, as the Oopsie recorder writes it. Other widths follow the profile's orientation representation.
- Gripper positions are scaled to [0, 1] with the minimum and maximum over all selected episodes of the same robot and gripper (`gripper: normalized`).
- A gripper key must have one value per arm, so episodes with `dexterous_hand` set are skipped. Recipes can leave them out up front with `selection.dexterous_hand: false` or `binary_gripper: true`.

Not unified: joint units (some labs record degrees), the gripper's open and closed direction, and robot kinematics.

### Rate

With `rate_hz` set, each output step takes the latest source step at or before it. Velocities keep their units, so they stay valid at the new rate. Videos are resampled to one frame per output step.

### Layout and masks

`layout.json` fixes the key order and the width of every key, which is the largest width any included episode has, or `pad_to`. The `action` and `state` vectors concatenate the keys in that order, padded with zeros. `action_mask` and `state_mask` mark the entries an episode actually has, so a 6-DoF arm and a 7-DoF arm, or a single arm and a bimanual setup, share one dataset.

## Formats

**`hdf5`**: the unified episodes themselves, together with `layout.json` and `manifest.parquet`.

| Location | Content |
|---|---|
| `episodes/<episode_key>.h5`: `action/<key>`, `state/<key>` | `(T, D)` float32, unpadded |
| `episodes/<episode_key>.h5` attrs | `fps`, `source_fps`, `num_steps`, `camera/<role>` (the video path, empty for a blank role) and `camera_sources` |
| `manifest.parquet` | every selected episode with `status` (`written`, `reused`, `skipped`, `failed`), `reason`, labels and widths |

**`lerobot`**: a [LeRobot](https://github.com/huggingface/lerobot) v2.1 dataset in `lerobot/`.
- Frames hold `observation.state`, `action`, their masks, the time and index columns, and one video feature `observation.images.<role>` per role.
- `meta/episodes.jsonl` also carries each episode's key, source, lab, robot, success and failure categories.
- Videos are hard links to the unified videos.
- LeRobot needs a single frame rate, so the recipe must set `rate_hz` unless every episode has the same rate.

**`rlds`**: a TFDS dataset at `rlds/<recipe name>/1.0.0`.
- Each step holds:
  - `observation/<role>` (JPEG), `observation/state` and `observation/<state key>`;
  - `action` and `action_dict/<key>`;
  - `reward` (1 on the last step of a successful episode), `discount`, `is_first`, `is_last`, `is_terminal` and `language_instruction`.
- `episode_metadata` holds the source, success, failure categories and masks.
- `file_path` starts with `success/` or `failure/`, from which `oopsie-rl` infers the outcome.
- Recipes that use the DROID camera names and keys produce the layout `oopsie-rl` reads.

To add a format, write `oopsie_backend/training/writers/<format>.py` with `write(episodes_dir, out, included, layout, recipe)`, and add it to `training/writers/registry.py` and `FORMATS` in `training/recipe.py`.
