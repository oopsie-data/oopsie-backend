# Configs

All settings live in `configs/`. They are edited by hand and take effect on the next run: each config is watched by the pipeline, so a change reruns the stages that read it ([what reruns](../pipeline/running.md#what-reruns)). Every file starts with a comment that explains its keys.

| File | Read by | Sets |
|---|---|---|
| `pipeline.yaml` | `oopsie pipeline run` | the [inputs](../pipeline/running.md#inputs) of every stage: paths, workers, Hub access |
| `repositories.yaml` | sync, index build, stats | the org's `archive` and `infra` repos; every other repo is a contributor repo |
| `imports.yaml` | sync, index build, [imports](../pipeline/imports.md) | the external datasets: Hub repos and local conversions |
| `lab_ids.yaml` | index build | canonical lab ids and the misspellings recorded for them |
| `camera_names.yaml` | [camera names](../pipeline/camera-names.md) | unified camera names and the aliases that map recorded names to them |
| `embodiments.yaml` | [embodiments](../pipeline/embodiments.md) | unified arms and end effectors, their types and aliases |
| `policies.yaml` | [policies](../pipeline/policies.md) | policy classes, families, frontier groups, aliases and patterns |
| `action_qa_config.yaml` | [QA](../pipeline/qa/index.md) | one section per check category: toggles, thresholds and lab feedback tags per check; sampling, scope and run defaults |
| `critical.yaml` | [critical episodes](../pipeline/critical-episodes.md) | the conditions and QA checks that make an episode critical |
| `stats_config.yaml` | [stats](../pipeline/stats.md) | placeholder lab ids and operator names, and the lab requirement thresholds |
| `eval_split.yaml` | [eval split](../pipeline/eval-split.md), [train sample](../pipeline/train-sample.md), [assignments](../pipeline/annotation/index.md#assignments) | the proposal search options, the chosen proposal, the train sample size, and the copies and seed of the assignments |
| `robot_aliases.yaml` | eval split, train sample | coarse robot platforms used to define scenes and common robots in split planning |
| `annotation_pipeline.yaml` | [annotation](../pipeline/annotation/index.md) commands and stages | the batch and returns repos, the annotation paths, and per batch its assignment, store and release folders |
| `annotation_flags.yaml` | [annotation flags](../pipeline/annotation/flags.md) | the note rules, which reasons exclude, and the idle thresholds |
| `analysis.yaml` | [dataset analysis](../pipeline/analysis.md) | the embedding model, similarity thresholds, verb synonyms and feature thresholds |
| `training/<recipe>.yaml` | [training data](../pipeline/training-data.md) | one training set: selection, unification and formats |
| `robot_profiles/` | converters | robot profiles that converters write into the episodes they produce |

## Changing a config safely

Most configs only change derived columns, and a change is undone by reverting it. Two need more care:

- **`eval_split.yaml`**: once a proposal is chosen, changing `propose_args`, `train_sample` or `assignments` makes the run fail rather than rewrite the frozen outputs ([human decisions](../pipeline/index.md#human-decisions-and-guarantees)).
- **`annotation_flags.yaml`**: a changed rule can exclude episodes from training and the statistics. Review the matches of every excluding reason after a change.

When adding a new dataset, expect issues from the unification stages: its camera, robot, gripper and policy names show up in `camera_name_issues`, `embodiment_issues` and `policy_issues` until they are added to the corresponding config.
