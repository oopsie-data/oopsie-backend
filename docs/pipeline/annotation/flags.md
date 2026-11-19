# Annotation flags

Annotators report problems with an episode in their notes and descriptions: a recording that is cut short, a person interfering with the robot, a broken video, or a request to delete the episode. The `annotation_flags` stage turns these reports into index columns, so that training and the statistics can leave out the affected episodes. The stage runs after `annotation_columns`, because it needs the collected annotations, and it never changes `critical` or the annotation records. Split planning therefore never sees the flags, and the eval split stays as it was chosen.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `annotation_flags` | `annotation` | `annotation_columns`, `episode_index`, `annotation_flags_config` | `annotation_flags`, `annotation_excluded`, `annotation_unusable` |

## Columns

- `annotation_flags`: one entry per `reason`, `annotator` and `source`, with the `text` that matched. The source is `additional_notes`, `episode_description` or `robot_state`.
- `annotation_excluded`: true when any flag's reason has `exclude: true`.
- `annotation_unusable`: the annotators whose record matched a rule with `unusable_record: true`.

All three columns are null for episodes without annotation records. Only each annotator's current record counts: their latest one, made on the episode's current content (`content_hash`).

## Reasons

The rules live in `configs/annotation_flags.yaml`. A note rule matches an annotator's `additional_notes` or `episode_description` with case-insensitive regular expressions. A match of one of its `unless` patterns in the same text cancels it, and one annotator's matching record is enough.

| Reason | Excludes | Matches |
|---|---|---|
| `delete_requested` | yes | the annotator asks to delete or discard the episode, unless the request is only because the video did not load |
| `cut_short` | yes | the recording ends before the attempt does |
| `human_interference` | yes | a person acts on the robot or the task objects during the attempt: intervenes, helps, touches the robot, moves or removes an object |
| `video_broken` | yes | frozen, corrupted, stitched or out-of-sync video, or very low resolution or frame rate |
| `human_in_scene` | no | a person is visible, sets up or resets the scene |
| `video_not_loaded` | no; the annotator's record is unusable | the annotator could not watch the video; the QA check `video/browser_playable` finds the cause |
| `teleoperated` | no | the annotator thinks the episode is teleoperated |
| `instruction_problem` | no | the instruction is vague, wrong or names objects that are not in the scene |
| `idle` | no | from the robot states (below) |

Exclusion follows the explicit request or the specific defect. Instruction problems are only flagged, because an instruction can be fixed without dropping the episode.

## Idle

`idle` reads the joint positions of the episode file, or the Cartesian positions when there are no joint states. A step is idle when the position changes slower than `idle_speed_fraction` (5%) of the episode's 95th percentile speed. The flag is raised when more than `max_idle_share` (60%) of the steps are idle, or when the robot is idle for at least `min_idle_start_seconds` (5 s) at the start and that is at least `min_idle_start_share` (25%) of the episode. Because speeds are relative to the episode's own motion, the flag finds long pauses but not episodes in which the robot only jitters.

## Use

Training curation leaves out every episode with `annotation_excluded`, in the eval splits as well as in `train_annotated` ([training data](../training-data.md)). A held-out eval episode is otherwise kept even when it becomes critical later, so the eval set changes only through these flags.

The [dataset analysis](../analysis.md) leaves out the excluded episodes from every group, and the annotator records in `annotation_unusable` from the agreement and label statistics. Nothing is deleted: the index keeps every record and flag, so changing a rule and rerunning the stage brings them back.

Rules are tuned on the collected notes. After changing them, rerun the command and review the matches of every excluding reason before relying on the new exclusions.

## Running it by hand

```bash
uv run oopsie annotation flags --data-dir <mirror> --index episode_index.parquet [--workers N]
```
