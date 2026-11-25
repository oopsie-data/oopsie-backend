# QA

Labs record with different robots, controllers and export scripts, and many of the problems in their data cannot be seen in the metadata: actions stored in degrees, velocities that are really positions, quaternions in the wrong order, cameras that freeze. QA reads each episode's actions, states and videos and runs a set of checks for such problems. Its findings feed the [critical flags](../critical-episodes.md) and the feedback the team sends to labs.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `qa_results` | `qa` | `duplicate_episodes`, `qa_check_versions` | `qa_*` columns; `reports/action_qa.html` |

## Checks

Each check belongs to a category and tests one property:

- **[Action checks](action-checks.md)**: `general` applies to every episode, and one category applies per declared action space (`joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`). They compare each action with the robot state it should lead to.
- **[Video checks](video-checks.md)**: the `video` category decodes every camera and compares the frames with the robot's motion.

Toggles and thresholds are in `configs/action_qa_config.yaml`, with one section per category and one entry per check.

A check either finds problems, records a skip with a reason when it cannot run on an episode (for example `no_matching_state`), or passes. Skips are stored, so a check that never applies to a dataset can be told apart from one that passes.

## What a run checks

QA checks every episode that is on disk and not a duplicate copy, from contributor and, by default, imported datasets. `sampling.default_episode_limit` and `sampling.video_episode_limit` can cap the number of episodes per dataset that get the action and video checks, for quick runs on large mirrors.

Results are stored per episode and check, together with the check's version. The version is a hash of the check's file, the shared logic it calls, its config section and, for video checks, the video settings. A run reads each episode once and runs only the checks whose stored result is missing or from an older version:

- adding a check runs only that check;
- changing a check's code or config section reruns only that check;
- deleting a check removes its results;
- disabling a check hides its results until it is enabled again.

A full run decodes every video and takes a long time, so a run saves its results to the index every five minutes (`run.save_interval`) and when it is stopped with SIGINT or SIGTERM. The next run continues with the remaining episodes. A progress line reports the episodes done, the rate and the estimated time left.

## Running it by hand

```bash
uv run oopsie qa run --data-dir <mirror> --index episode_index.parquet --workers 16
uv run oopsie qa run --check video/frozen --force       # rerun one check
```

| Option | Effect |
|---|---|
| `--check CATEGORY/TEST` | run only these checks; repeatable |
| `--force` | rerun checks even when their results are current |
| `--retry-errors` | rerun checks that failed with an error |
| `--max-episodes N` | stop after N episodes |
| `--limit`, `--video-limit` | override the per-dataset sampling |
| `--save-interval S` | seconds between saves |

## Results

The index stores every result in `qa_results`, and summarizes the results at the current check versions in `qa_checked`, `qa_findings`, `qa_skips`, `qa_flagged` and related columns ([column reference](../../reference/index-columns.md#qa)).

`reports/action_qa.html` shows the results as a grid with one row per dataset and one column per check, with a drilldown into every flagged episode and the values that tripped the check. `qa run` writes it after every run; `oopsie qa render-report` writes it from the index alone. To look at the flagged videos as well, serve the report with its evidence panel:

```bash
OOPSIE_QA_PASSWORD=... uv run oopsie qa serve-report --index episode_index.parquet --data-dir <mirror>
```

The server asks for the password on every request, since it serves the labs' videos.

## Lab feedback

Every check has two tags in its config entry that decide how its findings are reported to labs:

- `confidence`: `high_confidence` when a finding is a real problem, `lower_confidence` when the check is a heuristic that can be wrong.
- `action`: `fix_and_resubmit` when the lab should fix and re-upload, `confirm_and_discuss` when the finding needs a conversation, or `exclude` to leave it out of the feedback.

`oopsie qa render-lab-feedback --index episode_index.parquet --drafts <drafts.txt>` writes an "Automated checks" block per lab, on its own and appended to each lab's section of the hand-written review drafts.

## Spot checks

Automated checks cannot judge everything a person sees in a video. [Spot checks](spot-checks.md) sample episodes per lab into bundles for manual review, and merge the reviewers' notes back into the index.
