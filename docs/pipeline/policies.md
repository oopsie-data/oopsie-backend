# Policies

Episodes name the policy that produced them freely, from `teleop` to checkpoint paths such as `stack-cube/2026-09-29_20-36-22_stack-std-scratch_s123/model_4200.onnx`. `configs/policies.yaml` maps every recorded `policy_name` to a policy family and a policy class. The recorded name stays in the index.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `unified_policies` | `preqa` | `duplicate_episodes`, `policies_config` | `policy_family`, `policy_class`, `frontier_group`, `policy_issues` |

## Columns

| Column | Content |
|---|---|
| `policy_family` | the model, e.g. `pi0.5`, `act`, `diffusion_policy`; `other` for a learned policy of no listed family |
| `policy_class` | one of the classes below |
| `frontier_group` | the model line of a `frontier` policy, from `frontier_groups`; null for other classes |
| `policy_issues` | policy names without a mapping, names marked `unknown` and frontier families without a group |

## Classes

| Class | Episodes |
|---|---|
| `teleop` | a human drives the robot |
| `frontier` | a pretrained generalist model (pi0, pi0-FAST, pi0.5, MolmoAct, PaliGemma DROID policies, X-VLA, GR00T, SmolVLA, Cosmos Policy, DreamZero), as a released checkpoint or fine-tuned by the lab |
| `custom` | a policy the lab trained that is not a frontier model |
| `rl` | a policy trained or refined with reinforcement learning, including residual RL on a frontier model |
| `dagger` | rollouts with human corrections: DAgger, HG-DAgger, intervention rounds |
| `scripted` | no learned policy: planners, scripted primitives, random actions |
| `mixed` | episodes from several policies, or from a policy and a human |
| `unknown` | the recorded name does not identify the policy |

A frontier model refined with reinforcement learning or human corrections keeps its family but takes the class `rl` or `dagger`, so `policy_family == "pi0.5"` selects every pi0.5 policy.

`frontier_groups` sorts the frontier families into model lines:

| Group | Families |
|---|---|
| `pi` | `pi0`, `pi0_fast`, `pi0.5` |
| `molmoact` | `molmoact` |
| `paligemma_droid` | `paligemma`, the RoboArena DROID baselines |
| `nvidia` | `gr00t`, `cosmos_policy`, `dreamzero` |
| `xvla` | `xvla` |
| `smolvla` | `smolvla` |

A frontier family without a group is reported as an issue.

## Mapping

Each family has a class in `families`. A rule names a family, which takes that class, or `{family, class}` for another class, such as `{family: pi0.5, class: rl}`. A recorded name is looked up in this order:
1. `datasets.<dataset>.aliases`, then `datasets.<dataset>.patterns`: rules for one dataset, by repository name without its namespace. A duplicate copy uses the dataset of its original.
2. `aliases`: whole names.
3. A recorded name that is itself a family name, such as `pi0.5` or `act`.
4. `patterns`: case-insensitive regular expressions searched in the name; the first match wins.
5. `datasets.<dataset>.default`: for datasets whose remaining names are checkpoints of the lab's own policies.

The family `unknown` is reported as an issue, as are names no rule matches. When a new dataset arrives, its unmatched names show up in the command's output; add them to `configs/policies.yaml`.

## Running it by hand

```bash
uv run oopsie index policies --index episode_index.parquet
```
