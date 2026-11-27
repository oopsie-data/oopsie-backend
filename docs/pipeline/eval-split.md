# Eval split

The eval set measures how well a model trained on Oopsie-Data generalizes. It holds out episodes at three levels of novelty, which annotators then label in detail:

| Split | Held out | How it is chosen |
|---|---|---|
| `id` | episodes whose scene and task both remain in train | a stratified sample over `(lab, robot, cameras, task)` groups |
| `od_task_id_scene` | a task never seen in train, in a scene that is | whole tasks from medium-sized multi-task labs |
| `od_task_od_scene` | a lab never seen in train | whole small labs, on common and on custom robots |

The h5 files have no field for the scene, so a scene is approximated by `(lab, robot, cameras)`. A task is the normalized language instruction: lowercased, with whitespace collapsed and surrounding punctuation stripped. Episodes beyond what the annotation budget allows for a held-out unit go to a fourth group, `excluded`, which belongs to neither train nor eval. Episodes of a held-out lab or task that the chosen proposal does not list, such as later uploads, are recorded as `excluded` as well.

## In the pipeline

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `eval_proposals` | `proposals` | `critical_episodes`, `split_config`, `robot_aliases_config` | `<data_dir>/annotation/proposals/`: one JSON per proposal, `candidates.json`, `REPORT.md` |
| `chosen_proposal` | `splits` | `eval_proposals`; waits for `chosen_proposal` in `configs/eval_split.yaml` | `<data_dir>/annotation/eval_proposal.json`, written once |
| `eval_split` | `splits` | `chosen_proposal`, `episode_index` | `eval_split`, `eval_proposal` |

Choosing the split is a human decision:

1. `eval_proposals` generates several candidate splits from the current index and writes `REPORT.md`, which compares them.
2. Someone reviews the report, and possibly the eval episodes themselves ([below](#reviewing-a-proposal)), and sets `chosen_proposal` in `configs/eval_split.yaml` to a proposal id.
3. `chosen_proposal` freezes that proposal as `annotation/eval_proposal.json`, and `eval_split` records it in the index. The [train sample](train-sample.md) and the [annotation bundles](annotation/index.md#bundles-and-the-roster) follow.
4. The frozen proposal is copied into the backend repo, for example as `configs/eval_proposal.json`, and `chosen_proposal` is set to that path.

A proposal id only names a file in `annotation/proposals/`, and a `data_dir` without a frozen proposal regenerates those files from its own index, so the same id can name a different split there. A `chosen_proposal` ending in `.json` is instead a proposal file relative to the backend repo. The stage copies it to `annotation/eval_proposal.json`, and if that file already exists, it must hold the same proposal or the run fails. Proposals are not regenerated while `chosen_proposal` names a repo file.

From then on the split does not change. The frozen proposal is written once, and `eval_proposals` stops regenerating proposals, so later changes to QA or the critical flags cannot affect them ([guarantees](index.md#human-decisions-and-guarantees)).

`propose_args` in `configs/eval_split.yaml` holds the options the stage passes to the proposal search, described below. `--train-only LAB` there keeps a lab out of every eval split.

## How proposals are built

Proposals use only episodes that are on disk, public, readable, and neither a duplicate copy nor critical.

**Budget.** The annotation budget, 2,000 episodes by default, is divided among the three splits by `--split-weights`. Within a held-out split, each held-out lab or task gets an equal share as its target size. Episodes of one task are often near-identical, so every held-out task is labeled with at most `--holdout-task-cap` episodes, and the rest go to `excluded`.

**Candidates.** A candidate lab or task fits well when its size after the cap is close to its target and the cap removes few of its episodes. Candidates are ranked by `|log(capped / target)| + log(episodes / capped)`, which penalizes both. A held-out task must not appear in any other lab and must have no near-duplicate instruction anywhere in the dataset. Every scene it uses must keep training episodes on other tasks. "Medium" and "small" labs are defined by rank in the lab size ordering.

**Variety.** The proposals should offer real alternatives, so they are built one after another, and a candidate's score grows by 1 for each earlier proposal that used it. A held-out task's score also grows by 0.5 for each earlier use of its lab. By default, every lab's best task is chosen before any lab's second, so the held-out tasks spread over many labs.

Once the units are chosen, each split's budget is filled evenly across them, and units smaller than an even share pass their remainder to the larger ones.

## Reviewing a proposal

`REPORT.md` has a table of the labs, the candidate table, and one section per proposal with its statistics and checks:

- **Violations** break the split's definition: an `id` episode whose group has no train episodes, a held-out task that also appears in train, a held-out lab that leaks into train. The summary table counts them.
- **Notes** need a human look: robots that exist only in held-out labs, tasks that held-out labs share with train, and pinned tasks with a near-duplicate train task. A near-duplicate has a similarity of at least 0.85, the larger of the difflib ratio and the word-set Jaccard index.

Instructions are compared as strings, so a paraphrase with low string similarity is not caught. The nearest-task column of the candidate table helps with that manual pass.

To look through a proposal's eval episodes, package them on LS6:

```bash
uv run oopsie splits package \
  --proposal /corral/utexas/IRI26024/oopsiedata/annotation/proposals/proposal_01.json \
  --data-dir /corral/utexas/IRI26024/oopsiedata
```

This writes one zip per split to `annotation/review_bundles/proposal_01/`. Each zip unpacks to `<split>/episodes/`, ordered by lab and task, which `oopsie-data annotate --samples-dir <split>/episodes` opens. Unlike the annotation bundles, these copies are not blinded: they keep their annotations and source data, and `eval_manifest.json` lists each episode's source, lab, robot and task. `--only SPLIT ...` packages some splits only.

## Running it by hand

```bash
uv run oopsie splits propose --index episode_index.parquet --out-dir eval_split_proposals
uv run oopsie splits record --proposal eval_split_proposals/proposal_02.json --index episode_index.parquet
```

`record` writes a proposal's splits to the index, as the `eval_split` stage does.

### Proposal options

| Option | Default | Effect |
|---|---|---|
| `--eval-budget` / `--budget-frac` | 2000 / 0.05 | episodes to label in total; `--eval-budget -1` uses the fraction of the dataset instead |
| `--split-weights` | 0.2,0.4,0.4 | budget shares of `id`, `od_task_id_scene` and `od_task_od_scene` |
| `--num-proposals` | 4 | proposals to write |
| `--num-od-tasks` | 16 | held-out tasks |
| `--od-tasks-per-lab` | 1 | most held-out tasks from one lab |
| `--no-od-task-spread` | off | rank held-out tasks by score alone, instead of every lab's best task first |
| `--num-ood-common` / `--num-ood-custom` | 6 / 2 | held-out labs on common and on custom robots |
| `--ood-common LAB` / `--ood-custom LAB` | search | pin held-out labs |
| `--od-task 'LAB::instruction'` | search | pin held-out tasks; repeated pins form one set that replaces the search |
| `--train-only LAB` | none | keep a lab out of every eval split |
| `--common-robot` / `--custom-robot` | auto | override a robot's class; a robot is common when at least two labs use it |
| `--ood-lab-min` | 40 | smallest held-out lab |
| `--ood-lab-max-quantile` | 0.3 | held-out labs come from the smallest 30% of labs |
| `--od-task-lab-quantiles` | 0.25,0.75 | size-rank window of labs that may lose a task |
| `--od-task-min-tasks` / `--od-task-min-episodes` | 3 / 25 | such a lab needs this many tasks with this many episodes each |
| `--od-task-max-share` | 0.4 | largest share of its lab a held-out task may hold |
| `--id-alpha` | 0 | `id` weight of a lab or group is `size**alpha`: 1 is proportional, 0 uniform |
| `--id-max-group-frac` | 0.25 | largest fraction of a group taken for `id` |
| `--id-min-group` | 8 | groups smaller than this are never sampled for `id` |
| `--holdout-task-cap` | 50 | most labeled episodes per held-out task; 0 disables the cap |
| `--no-holdout-cap` | off | label held-out units in full |
| `--seed` | 0 | sampling seed |
