# Embodiments

Episodes name their robot and gripper freely: the same arm appears as `franka_research_3`, `franka_fr3` and `Franka Research 3`, and some names describe a whole setup (`ARX X5 dual-arm manipulator mounted on a mobile base`). `configs/embodiments.yaml` maps every recorded `robot_name` to a unified arm and every `gripper_name` to a unified end effector.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `unified_embodiments` | `preqa` | `duplicate_episodes`, `embodiments_config` | `unified_arm`, `unified_end_effector`, `end_effector_type`, `embodiment`, `embodiment_issues` |

## Columns

| Column | Content |
|---|---|
| `unified_arm` | the arm model, e.g. `franka_fr3`; two models joined by `+` (left, right) for a mixed bimanual setup |
| `unified_end_effector` | the end effector model, e.g. `robotiq_2f_85` |
| `end_effector_type` | `parallel_jaw`, `soft`, `dexterous_hand`, `tool` or `other` |
| `embodiment` | `<layout>/<arm>/<end effector>`, e.g. `bimanual/yam/yam_gripper` |
| `embodiment_issues` | recorded names without a mapping and names marked `unknown` |

The layout is `single_arm` or `bimanual`, from the index's `biarm` column, prefixed with `mobile_` when `uses_mobile_base` is set. Arm names therefore name only the arm model: `dual_ufactory_xarm7` and `ufactory_xarm7` are both `xarm7`. `embodiment` is null unless both arm and end effector are named.

`end_effector_type` describes the hardware. The index's `dexterous_hand` column describes the recorded actions instead, so a hand driven by a single grip value is a `dexterous_hand` end effector, while its `dexterous_hand` column is false.

## Mapping

A recorded robot name is looked up in this order:
1. `datasets.<dataset>.arm_aliases`: names for one dataset, by repository name without its namespace. A duplicate copy uses the dataset of its original.
2. `arm_aliases`: names for every dataset.
3. `arms`: a recorded name that is itself a unified arm name maps to itself.

Gripper names follow the same order with `end_effector_aliases` and `end_effectors`, which also gives each end effector its type. A mapping to `unknown` reports the name as an issue: the hardware is known to exist, but which model it is has not been settled.

When a new dataset arrives, its names show up as issues in the command's output; add them to `configs/embodiments.yaml`.

## Running it by hand

```bash
uv run oopsie index embodiments --index episode_index.parquet
```
