# Critical episodes

Some episodes cannot be used at all: the file is unreadable, the actions do not line up with the states, or a camera is missing. The critical stage collects these conditions into a single flag, so that the stats, split planning and training can leave such episodes out without knowing the details.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `critical_episodes` | `postqa` | `qa_results`, `duplicate_videos`, `critical_config` | `critical`, `critical_reasons` |

## Conditions

`configs/critical.yaml` lists the conditions. An episode is critical when any of them holds, and `critical_reasons` names each one that does:

| Reason | Condition |
|---|---|
| `read_error` | the h5 could not be read |
| `duplicate_video` | a video file is shared with another camera or episode ([duplicates](duplicates.md#duplicate-videos)) |
| `qa_error` | a QA check failed with an error; off by default |
| `short` | the episode is shorter than `min_duration_seconds` (2 s) |
| `qa:<category>/<test>` | a finding of a QA check listed under `qa_checks`, or, for checks under `qa_check_min_findings`, at least the given number of findings in one video |

The default list holds the checks whose findings make an episode structurally unusable: mismatched action and state lengths, action chunks stored per step, identical action arrays, missing required states, a robot that never moves, wrong joint counts, malformed quaternions or velocity vectors, undecodable videos, video that freezes while the robot moves, and repeatedly corrupted frames.

Only episodes that QA has checked can be critical through a QA check. Duplicate copies are never critical, because every later stage skips them anyway.

## Effect

The stats, split planning, the train sample, training and the dataset analysis leave critical episodes out. One exception protects the eval set: training keeps held-out eval episodes even if they become critical after the split was chosen ([guarantees](index.md#human-decisions-and-guarantees)).

The internal stats report lists the critical episodes per lab with their reasons.

```bash
uv run python -m oopsie_backend.postqa.critical --index episode_index.parquet
```
