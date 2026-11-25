# Training dataset preparation

Build training datasets from indexed Oopsie episodes using YAML recipes:

`recipe → select episodes → convert/resample arrays and cameras → export formats`

Recipes control episode filters, action/state representations, rates, cameras,
padding and output formats. Start with one in [configs/training](../../configs/training):

| Recipe | Output |
|---|---|
| `droid_joint_velocity.yaml` | Single-arm Franka, joint velocity + gripper, 15 Hz, DROID camera keys; RLDS |
| `eef_quat.yaml` | Single-arm end-effector pose + gripper, quaternion orientation, 10 Hz; LeRobot |
| `all_native.yaml` | Each episode's native keys, 10 Hz, padding/masks; HDF5 and LeRobot |

## Run a recipe

Run commands from the repository root after `uv sync`. Direct builds need a local
data mirror and an episode index with unified camera names already populated:

```bash
uv run oopsie training materialize \
  --recipe configs/training/droid_joint_velocity.yaml \
  --data-dir /path/to/mirror \
  --index /path/to/episode_index.parquet \
  --out /path/to/training
```

The CLI remains `training materialize`. The equivalent module entry point is
`uv run python -m oopsie_backend.training.build_dataset` with the same arguments.
Use `uv run oopsie training materialize --help` for all options.

To refresh upstream pipeline stages before building, use the configured paths in
`configs/pipeline.yaml`:

```bash
uv run oopsie pipeline run training \
  --set training_recipe=configs/training/droid_joint_velocity.yaml
```

On Slurm: `sbatch slurm/pipeline.slurm training --set training_recipe=configs/training/droid_joint_velocity.yaml`.
Training is manual; the nightly pipeline does not run it.

Optional direct-build flags: `--workers N` sets parallelism, `--work-dir PATH`
keeps reusable prepared episodes on scratch storage, and `--stage-dir PATH`
copies source episodes there in temporary chunks before processing.

## Files and responsibilities

Only `build_dataset.py` is a standalone command. Other modules are imported helpers;
`pipeline_stages.py` registers stages when the pipeline loads.

| File | Responsibility / main API |
|---|---|
| [build_dataset.py](build_dataset.py) | Coordinates selection, preparation, export and index updates. `materialize(...)` or CLI `main(...)`. |
| [recipe.py](recipe.py) | Recipe defaults, validation and camera presets. `load_recipe(path)`, `unify_hash(recipe)`. |
| [select_episodes.py](select_episodes.py) | Index filters, annotation labels and deterministic sampling. `select_episodes(index, recipe)`. |
| [read_episode.py](read_episode.py) | Reads source arrays/profile/video paths and validates lengths/rate. `read_source(path, repo_path)`, `SourceEpisode`, `Skip`. |
| [convert_actions_states.py](convert_actions_states.py) | Action derivation, pose conversion, gripper normalization and actions from states. `Unifier`, `action_from_state(...)`. |
| [prepare_episode.py](prepare_episode.py) | Converts/resamples one episode, writes arrays/videos, reuses current outputs and reports status. `unify_episode(job)`. |
| [resample.py](resample.py) | Target-rate sampling, interpolation/holding and filled-step flags. `plan(...)`, `resample(...)`. |
| [cameras.py](cameras.py) | Role matching, fallbacks, missing-view rules, resizing and video encoding. `assign_roles(...)`, `transcode(...)`. |
| [vector_layout.py](vector_layout.py) | Shared key order, widths, zero padding and masks. `build_layout(...)`, `padded(...)`. |
| [pipeline_stages.py](pipeline_stages.py) | Manual pipeline registration and upstream dependencies. `training_recipe_state`, `training_data`. |
| [writers/registry.py](writers/registry.py) | Lazily resolves formats to `write(...)`; HDF5 needs no separate writer. |
| [writers/rlds.py](writers/rlds.py) | RLDS/TFDS images, actions/states, masks, language, rewards, flags and provenance. |
| [writers/lerobot.py](writers/lerobot.py) | LeRobot v2.1 Parquet, videos, metadata and episode statistics. |

For Python callers, use `materialize(recipe_path, data_dir, index, out, work_dir,
workers, stage_dir=None)` from `oopsie_backend.training.build_dataset` (paths are
`pathlib.Path` objects). Format writers share
`write(episodes_dir, out, included, layout, recipe)`.

## Outputs and reruns

Each dataset lives in `<out>/<recipe name>/`:

| Path | Contents |
|---|---|
| `manifest.parquet` | Every selected episode's labels, status, dimensions and skip/failure reason. |
| `layout.json` | Action/state key order, padded widths, camera roles, image size and common rate if any. |
| `recipe.yaml` | Resolved recipe, including defaults. |
| `episodes/` | Prepared HDF5 arrays and per-role videos; retained in work storage and copied to output for `hdf5`. |
| `rlds/<recipe name>/1.0.0/` | TFDS dataset when `formats` includes `rlds`. |
| `lerobot/` | LeRobot dataset when `formats` includes `lerobot`. |

The build updates the index's `training_sets` column. Reruns reuse prepared
episodes when source metadata, conversion settings/code, camera names and gripper
ranges match, then rebuild requested format outputs. Changing labels alone does
not require preparing episodes again. A different work directory is copied to
the output after completion, replacing the previous dataset for that recipe.

Camera roles require unified names from `oopsie index camera-names`. LeRobot
requires a common rate; RLDS currently writes one TFDS `train` split from the
recipe's selected episodes. Joint units, gripper open/closed direction and robot
kinematics are not standardized here.

See [the full training guide](../../docs/pipeline/training-data.md) for recipe settings,
conversion rules and format schemas.
