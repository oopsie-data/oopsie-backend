# Index columns

Every column of `episode_index.parquet`, grouped by the stage that writes it. Each stage writes only its own columns. A column is null for episodes the stage has not processed, for example QA columns of episodes QA has not checked, or annotation columns of episodes nobody annotated.

## Core

Written by the [index build](../pipeline/episode-index.md) (`episode_index`).

| Column | Type | Content |
|---|---|---|
| `episode_key` | string | the row key: the first 16 hex characters of `sha1("<dataset_repo>/<repo_path>")` |
| `dataset_repo` | string | the repo, e.g. `OopsieData-Submissions/SomeLab`, or `imports/<name>` for a local import |
| `repo_path` | string | the h5 path inside the repo |
| `repo_kind` | string | `contributor` or `external` ([repo kinds](../architecture.md#repos)) |
| `public` | bool | counted in the public stats: a contributor repo whose lab id is not a placeholder |
| `on_disk`, `on_hub` | bool | whether the h5 is in the mirror and in the Hub listing; rows whose file disappeared stay with `on_disk = false` |
| `schema`, `episode_id`, `operator_name`, `language_instruction` | string | root attributes of the h5 |
| `lab_id` | string | the lab id, mapped to its canonical name by `configs/lab_ids.yaml` |
| `timestamp` | datetime | the recording time; null when missing or before 2000 |
| `task` | string | `language_instruction` lowercased, with whitespace collapsed and surrounding punctuation stripped |
| `robot_name`, `gripper_name`, `policy_name` | string | from the robot profile, as recorded |
| `control_freq` | float | control frequency in Hz |
| `is_biarm`, `uses_mobile_base` | bool | from the robot profile |
| `single_arm`, `biarm` | bool | one arm or two, from `is_biarm` |
| `action_space`, `robot_state_keys` | list of strings | the declared action keys and the recorded state keys |
| `gripper_dof` | int | values per arm of the gripper action (`gripper_position`, else `gripper_velocity`, else `gripper_binary`) |
| `binary_gripper` | bool | the episode has a `gripper_binary` action |
| `dexterous_hand` | bool | more than one gripper value per arm, i.e. a hand |
| `cameras` | list of strings | the recorded camera names |
| `num_steps`, `duration_seconds` | int, float | length of the first non-empty action array, and that over `control_freq` |
| `annotations` | list of structs | the lab's own `episode_annotations`: name, timestamp, success, outcome, failure categories, severity, description, notes |
| `content_hash` | string | sha256 over all action and state arrays; identifies the recorded data independently of the file |
| `h5_size`, `h5_mtime_ns` | int | local file size and modification time, which decide whether the file is reread |
| `h5_sha256`, `h5_git_oid` | string | the Hub's LFS sha256 and git object id of the h5 |
| `videos` | list of structs | per camera: repo path, local size, Hub sha256 and git object id |
| `read_error` | string | why the h5 could not be read; the h5 columns are then null |
| `indexed_at` | datetime | when the h5 was last read |

## Upload times

Written by `hub_upload_times` ([Hub sync](../pipeline/sync.md#upload-times)).

| Column | Type | Content |
|---|---|---|
| `hub_uploaded_at` | datetime | date of the commit that first added the h5 to the repo's default branch |
| `hub_upload_commit` | string | that commit |

## Duplicates

Written by `duplicate_episodes` and `duplicate_videos` ([duplicates](../pipeline/duplicates.md)).

| Column | Type | Content |
|---|---|---|
| `duplicate_of` | string | the key of the kept copy; null for kept episodes |
| `duplicate_reasons` | list of strings | the rules linking the episode to other copies: `identical_data`, `identical_videos`, `same_metadata` |
| `duplicate_group` | list of strings | the keys of the other copies |
| `video_duplicates` | list of structs | per camera whose file is shared: the hash and every other episode and camera with it |
| `has_duplicate_video` | bool | any camera's file is shared |

## Unification

Written by `unified_camera_names`, `unified_embodiments` and `unified_policies`.

| Column | Type | Content |
|---|---|---|
| `unified_cameras` | list of structs | per recorded camera, its unified `name`, or null when it is left out ([camera names](../pipeline/camera-names.md)) |
| `camera_name_issues` | list of strings | unmapped cameras, cameras marked `unknown`, names two cameras share |
| `unified_arm`, `unified_end_effector` | string | the arm and end effector model ([embodiments](../pipeline/embodiments.md)) |
| `end_effector_type` | string | `parallel_jaw`, `soft`, `dexterous_hand`, `tool` or `other` |
| `embodiment` | string | `<layout>/<arm>/<end effector>` |
| `embodiment_issues` | list of strings | unmapped names and names marked `unknown` |
| `policy_family`, `policy_class` | string | the policy's model family and class ([policies](../pipeline/policies.md)) |
| `frontier_group` | string | the model line of a `frontier` policy |
| `policy_issues` | list of strings | unmapped names, names marked `unknown`, frontier families without a group |

## QA

Written by `qa_results` ([QA](../pipeline/qa/index.md)). The summary columns cover only results at the current check versions.

| Column | Type | Content |
|---|---|---|
| `qa_results` | list of structs | one entry per check that ran: check, version, run time, whether it applied, error, findings and skips |
| `qa_run_at` | datetime | the latest run on the episode |
| `qa_error` | string | errors of checks that failed |
| `qa_video_checked` | bool | the video checks ran |
| `qa_checked` | list of strings | `category/test` of every check that applied |
| `qa_findings` | list of structs | category, test, key (e.g. the camera) and JSON details of each finding |
| `qa_skips` | list of structs | category, test, key and reason of each skip |
| `qa_flagged` | bool | any finding |
| `qa_notes` | list of structs | notes from [spot checks](../pipeline/qa/spot-checks.md): bundle, QA id, reviewer, time and text; written by `oopsie qa merge-notes --index` |

## Critical

Written by `critical_episodes` ([critical episodes](../pipeline/critical-episodes.md)).

| Column | Type | Content |
|---|---|---|
| `critical` | bool | the episode is structurally unusable |
| `critical_reasons` | list of strings | `read_error`, `duplicate_video`, `qa_error`, `short` or `qa:<category>/<test>` |

## Eval split

Written by `eval_split` ([eval split](../pipeline/eval-split.md)).

| Column | Type | Content |
|---|---|---|
| `eval_split` | string | `id`, `od_task_id_scene`, `od_task_od_scene` or `excluded`; null for train episodes |
| `eval_proposal` | string | the proposal the split came from |

## Annotation

Written by `annotation_columns` ([annotation](../pipeline/annotation/index.md#index)).

| Column | Type | Content |
|---|---|---|
| `annotation_batch` | string | the batch the episode was assigned in |
| `annotation_split` | string | its split in that batch, including `train_annotated` |
| `annotation_assigned` | list of strings | the annotators it was assigned to |
| `annotation_returned` | list of strings | the annotators who returned it |
| `annotation_records` | list of structs | every collected annotation, described below |

Each record holds:

- `annotator`, and the `name` and `timestamp` of the annotation;
- the labels `outcome`, `failure_category`, `severity`, `episode_description` and `additional_notes`;
- `temporal_segments`, each with `start` and `end` as inclusive frame indices, `type`, `label` and `failure_category`;
- `content_hash`, the hash of the copy the annotator labeled; the record applies while it equals the row's `content_hash`;
- `language_instruction`, the annotator's corrected instruction where it differs from the episode's, else null.

## Annotation flags

Written by `annotation_flags` ([annotation flags](../pipeline/annotation/flags.md)).

| Column | Type | Content |
|---|---|---|
| `annotation_flags` | list of structs | one entry per reason, annotator and source (`additional_notes`, `episode_description` or `robot_state`), with the matching text |
| `annotation_excluded` | bool | a flag's reason excludes the episode |
| `annotation_unusable` | list of strings | annotators whose record the statistics leave out |

## Training

Written by `training_data` ([training data](../pipeline/training-data.md)).

| Column | Type | Content |
|---|---|---|
| `training_sets` | list of strings | the recipes whose materialized dataset holds the episode |
