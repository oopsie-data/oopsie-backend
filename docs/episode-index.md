# Episode index

`episode_index.parquet` holds one row per episode, keyed by `episode_key`. That key is the first 16 hex characters of `sha1("<dataset_repo>/<repo_path>")`. Every pipeline step reads its episodes from the index and writes its per-episode results back as its own columns. On LS6 the index lives at `/corral/utexas/IRI26024/episode_index/episode_index.parquet`, and the nightly job refreshes its core columns after the Hub sync.

```python
import polars as pl
df = pl.read_parquet("episode_index.parquet")
contributor = df.filter(pl.col("on_disk") & pl.col("public") & pl.col("duplicate_of").is_null())
```

## Browse as HTML

Create a standalone viewer and open the resulting HTML file in a browser:

```bash
uv run oopsie tools parquet-html data/releases/2026-11-08/episode_index_sample.parquet
```

The HTML is written beside the source with the same name and an `.html` extension.
It works offline and provides search, sorting, pagination, visible-column controls,
column types and a row inspector for nested lists, annotations and QA results.
It does not load episode videos or modify the Parquet file.

By default it embeds the first 1,000 rows and labels the preview with the full row
count. Search and sorting operate on embedded rows only. Use `--max-rows 0` to
embed all rows (large indexes can produce large HTML files), `--columns NAME ...`
to export selected columns, or `--out PATH.html` to choose the output path:

```bash
uv run oopsie tools parquet-html data/releases/2026-11-08/episode_index.parquet \
  --max-rows 0 --columns episode_key dataset_repo robot_name language_instruction \
  --out reports/episode_index.html
```

Timestamps use ISO strings; integers beyond JavaScript's exact integer range,
decimals and nonfinite numbers are displayed as strings to preserve their values.

## Download the files in an index

Use `oopsie ingest download` with an episode-index Parquet or a filtered copy:

```bash
uv run oopsie ingest download \
  --index data/releases/2026-11-08/episode_index_sample.parquet \
  --data-dir data/releases/2026-11-08/sample_data \
  --revisions data/releases/2026-11-08/snapshot.json
```

The Parquet needs `dataset_repo` (`namespace/repo`) and `repo_path` (a file path
inside the repo). If it has a `videos` list with a `path` field per video, the
command downloads those files too. Files go under `<data-dir>/<namespace>/<repo>/`
with their original paths. The command skips local imports, verifies file
hashes, and reuses matching files on reruns. It uses your Hugging Face login
or `HF_TOKEN` for private datasets.

Pass `--revisions` with a release's `snapshot.json` to use its pinned commits.
Without a snapshot, the command requests files from the repos' default branches.

## Writing

`oopsie_backend.index.store` reads and writes the index:

- `update_columns(path, frame, columns, clear_missing=False)` sets a step's columns for the episodes in `frame` and leaves all other columns untouched.
- With `clear_missing=True`, the step's columns are reset to null for every episode not in `frame`. Use this for steps that recompute their full output.
- Every write holds an exclusive lock on `<index>.lock` and replaces the file atomically, so steps running at the same time never lose each other's columns.
- Keys that are not in the index are an error.

## Repos

Episodes are discovered under `<data_dir>/OopsieData-Submissions/<repo>/` and, for local imports, under `<data_dir>/imports/<name>/` with the dataset id `imports/<name>`. `configs/repositories.yaml` lists the archive and infra repos and `configs/imports.yaml` the imported datasets (see [imports](imports.md)); every other repo is a contributor repo:

| `repo_kind` | Repos | Indexed |
|---|---|---|
| `contributor` | every org repo not listed | yes, `public` unless the lab id is a placeholder |
| `external` | every entry of `imports.yaml`: org repos (SOAR, AutoEval, RoboArena, AI2 MolmoAct) and local imports | yes, never `public`; local imports have no Hub hashes |
| `archive` | `archive` | no |
| `infra` | `infra`, including the annotation repo | no |

## Columns

### Core (`index.build`)

| Column | Content |
|---|---|
| `episode_key`, `dataset_repo`, `repo_path` | identity; `repo_path` is the h5 path inside the HF repo |
| `repo_kind`, `public` | see above |
| `on_disk`, `on_hub` | whether the h5 is in the local mirror and in the Hub listing; rows whose file disappeared stay with `on_disk = false` |
| `schema`, `episode_id`, `lab_id`, `operator_name`, `language_instruction`, `timestamp` | root attributes of the h5; `timestamp` is null before 2000 |
| `task` | `language_instruction` lowercased, whitespace collapsed, surrounding punctuation stripped |
| `robot_name`, `gripper_name`, `policy_name`, `control_freq`, `is_biarm`, `uses_mobile_base`, `action_space`, `robot_state_keys` | from `robot_profile` |
| `single_arm`, `biarm` | embodiment: one arm or two, from `is_biarm` |
| `gripper_dof` | values per arm of the gripper action (`gripper_position`, else `gripper_velocity`, else `gripper_binary`) |
| `binary_gripper`, `dexterous_hand` | embodiment: a `gripper_binary` action or one gripper value per arm; more than one gripper value per arm (a hand) |
| `cameras` | keys of `observations/video_paths` |
| `num_steps`, `duration_seconds` | length of the first non-empty action array, and that over `control_freq` |
| `annotations` | the lab's `episode_annotations`, decoded with the annotation schema: name, timestamp, success, outcome, failure categories, severity, description, notes |
| `content_hash` | sha256 over all action and robot state arrays |
| `h5_size`, `h5_mtime_ns` | local file size and modification time |
| `h5_sha256`, `h5_git_oid` | from the Hub listing: LFS sha256 and git object id |
| `videos` | per camera: repo path, local size, Hub sha256 and git object id |
| `read_error` | why the h5 could not be read; the h5 columns are then null |
| `indexed_at` | when the h5 was last read |

An h5 is re-read only when its size or modification time changed, or when the index lacks one of the h5 columns, which happens once after a column is added. When the Hub listing of a repo fails, the earlier hashes and `on_hub` values are kept. `--no-hub` skips the listing.

### Hub upload times (`ingest.upload_times`)

`hub_uploaded_at` is the date of the commit that first added the episode's h5 to the default branch of its Hub repo, and `hub_upload_commit` is that commit. A file added by a merged pull request counts from the merge commit. Both are null for episodes not on the Hub. The history of each repo, commits and trees without file contents, is kept as a bare git clone in `<data root>/.hub_history/<repo>.git` and fetched incrementally. A repo whose fetch fails keeps its earlier values. A repo whose history was squashed on the Hub dates every file to the squash commit.

No stage depends on these columns, so they rerun after the sync or the index changes without invalidating anything downstream. `uv run oopsie ingest upload-times --data-dir <mirror> --index episode_index.parquet` runs the step on its own.

### Camera names (`index.camera_names`)

`unified_cameras` (`camera`, `name` per recorded camera; `name` is null for a camera that is left out) and `camera_name_issues` (cameras without a unified name, cameras marked `unknown`, and names two cameras of an episode share), as configured in `configs/camera_names.yaml`; see [camera names](camera-names.md).

### Duplicate episodes (`preqa.duplicate_episodes`)

Two on-disk episodes are copies when they have the same `content_hash` (`identical_data`), the same Hub hash for every camera's video (`identical_videos`), or the same lab, timestamp, duration, instruction, robot and cameras with a recorded timestamp (`same_metadata`, which catches re-exported or trimmed copies). Copies linked by any rule form a group. The first copy is kept, contributor repos first and then by repo and path; every other copy gets `duplicate_of`, its key. `duplicate_reasons` lists the rules that link an episode to another copy, and `duplicate_group` lists the other copies' keys. Split planning, the train sample, QA and the duplicate-video check leave out episodes with `duplicate_of` set.

### Duplicate videos (`preqa.duplicate_videos`)

`video_duplicates` lists, per camera whose file is identical to another video, the shared hash and every other episode and camera with it. `has_duplicate_video` summarizes it. Videos are compared by their Hub hash; episodes marked `duplicate_of` are left out.

### QA (`qa.run`)

`qa_results` holds one entry per check that ran: check name, version, run time, whether it applied, error, findings and skips. The summary columns cover the entries at the current check version:

- `qa_run_at`, `qa_error` and `qa_video_checked`
- `qa_checked`: `category/test` of every check that applied
- `qa_findings`: category, test, key and JSON details
- `qa_skips`
- `qa_flagged`

See [action checks](action-checks.md#running-and-adding-checks).

### Critical episodes (`postqa.critical`)

`critical` and `critical_reasons` (`read_error`, `duplicate_video`, `qa_error`, `short` or `qa:<category>/<test>`), as configured in `configs/critical.yaml`. `short` marks episodes shorter than `min_duration_seconds`. Checks under `qa_checks` make an episode critical on any finding; checks under `qa_check_min_findings` only when one video has at least the given number of findings.

### Manual QA notes (`qa.merge_notes --index`)

`qa_notes`: bundle, QA id, reviewer, update time and note text per returned note.

### Eval split (`splits.record`)

`eval_split` (`id`, `od_task_id_scene`, `od_task_od_scene` or `excluded`) and `eval_proposal`. A null `eval_split` means the episode is not held out.

### Annotation batches (`annotation.pipeline collect` or `index`)

`annotation_batch`, `annotation_split` (the split it was assigned in, including `train_annotated`), `annotation_assigned`, `annotation_returned` and `annotation_records` (the current collected annotation per annotator).

### Training data (`training.build_dataset`)

`training_sets`: the recipes whose materialized dataset holds the episode (see [training data](training-data.md)). Materializing a recipe replaces that recipe's entries.

## Commands

The [pipeline](pipeline.md) runs the index, pre-QA, QA and post-QA stages; each also runs on its own:

```bash
uv run oopsie index build --data-dir <mirror> --index episode_index.parquet
uv run oopsie qa run --data-dir <mirror> --index episode_index.parquet
uv run python -m oopsie_backend.preqa.duplicate_episodes --index episode_index.parquet
uv run python -m oopsie_backend.preqa.duplicate_videos --index episode_index.parquet
uv run python -m oopsie_backend.postqa.critical --index episode_index.parquet
uv run oopsie qa merge-notes returned/*.json --index episode_index.parquet
uv run oopsie splits record --proposal eval_split_proposals/proposal_02.json --index episode_index.parquet
uv run oopsie annotation pipeline collect
```

`oopsie splits propose` and `oopsie splits sample-train` accept the parquet index as `--index`. They use episodes that are on disk, public and readable, and neither a duplicate copy nor critical.
