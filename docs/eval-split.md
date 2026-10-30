# Eval split proposals

The eval set holds out three groups of episodes:

| Split | Definition | How it is chosen |
|---|---|---|
| `id` | Scene and task both present in train | Stratified sample over `(lab, robot, cameras, task)` groups |
| `od_task_id_scene` | Novel task, scene present in train | Full tasks from several medium multi-task labs, one task per lab |
| `od_task_od_scene` | Novel lab | Full small labs, on common and on custom robots |

A scene is approximated by `(lab, robot, cameras)`, because the h5 files have no background field. A task is the normalized language instruction: lowercased, with whitespace collapsed and surrounding punctuation stripped.

## 1. Get the index

The proposals are built from the [episode index](episode-index.md), which the nightly job refreshes on LS6 at `/corral/utexas/IRI26024/episode_index/episode_index.parquet`. Copy it to a laptop to run the next steps there. Only episodes that are on disk, public and readable, and neither a duplicate copy nor critical, are used. Index files in the earlier JSONL format are still read.

## 2. Generate proposals

```bash
uv run python -m oopsie_backend.splits.propose \
  --index episode_index.parquet --out-dir eval_split_proposals
```

Output:

```
eval_split_proposals/
  REPORT.md          # lab table, candidates, one section per proposal
  candidates.json
  proposal_01.json   # config, stats, checks, episode lists per split
  ...
```

An annotation budget, by default 2000 episodes, is split across the three groups by `--split-weights`. Each held-out lab and each held-out task gets an equal part of its group's budget as a target size, and a held-out task's target is at most `--holdout-task-cap`.

Every task in both held-out groups is labeled with at most `--holdout-task-cap` episodes, since more episodes of one task are mostly near-identical. A lab's labelable size is its episode count after this cap. Lab candidates are ranked by `|log(labelable / target)| + log(episodes / labelable)`, where the second term penalizes labs that would lose most of their episodes to the cap. Task candidates are ranked the same way, by `|log(capped / target)| + log(episodes / capped)`, where `capped` is the task's episode count after the cap. Once the units are chosen, each group's budget is filled evenly across its units, and units smaller than an even share pass their remainder to the larger ones. Episodes beyond a unit's share are moved to `excluded`, which is neither train nor eval.

"Medium" and "small" labs are defined by rank in the lab size ordering, not by share of the data. Task candidates must have no near-duplicate anywhere in the dataset. The candidate table lists each lab's best task before any lab's second, so its top rows span several labs.

The script writes `--num-proposals` (default 4) proposals that differ as much as the candidates allow. They are built one after another, and a candidate's score grows by 1 for every earlier proposal that used it, so unused labs and tasks win unless they fit much worse. A held-out task's score also grows by 0.5 for every earlier use of its lab, because held-out tasks come from a limited set of medium labs. Repeated pins of one kind form a single set that replaces the search for that kind. For example, `--od-task 'A::x' --od-task 'B::y'` holds out both tasks.

| Flag | Default | Meaning |
|---|---|---|
| `--eval-budget` / `--budget-frac` | 2000 / 0.05 | Episodes to label in total; pass `--eval-budget -1` to use the fraction instead |
| `--split-weights` | 0.2,0.4,0.4 | Budget shares of `id`, `od_task_id_scene` and `od_task_od_scene` |
| `--num-od-tasks` | 16 | Held-out tasks |
| `--od-tasks-per-lab` | 1 | Max held-out tasks from one lab; every lab's best task is chosen before any lab's second |
| `--no-od-task-spread` | off | Rank held-out tasks by score alone instead of choosing every lab's best task before any lab's second |
| `--num-ood-common` / `--num-ood-custom` | 6 / 2 | Held-out labs per embodiment class |
| `--ood-common LAB` / `--ood-custom LAB` | search | Pin held-out labs |
| `--od-task 'LAB::instruction'` | search | Pin held-out tasks |
| `--train-only LAB` | none | Lab that never enters an eval split, neither held out nor sampled for ID |
| `--common-robot` / `--custom-robot` | auto | Override robot class. Auto means common if at least two labs use it, after lowercasing and replacing punctuation |
| `--ood-lab-min` | 40 | Minimum size of a held-out lab |
| `--ood-lab-max-quantile` | 0.3 | Held-out labs come from the smallest 30% of labs |
| `--od-task-lab-quantiles` | 0.25,0.75 | Size-rank window of labs that may lose a task |
| `--od-task-min-tasks` / `--od-task-min-episodes` | 3 / 25 | Such a lab needs this many tasks with at least this many episodes |
| `--od-task-max-share` | 0.4 | A held-out task may hold at most this share of its lab |
| `--id-alpha` | 0 | ID lab and group weight is `size**alpha`; 1 is proportional, 0 is uniform. The ID target is split across labs first, then across each lab's groups |
| `--id-max-group-frac` | 0.25 | Max fraction of a group taken for ID eval |
| `--id-min-group` | 8 | Groups smaller than this are never sampled |
| `--holdout-task-cap` | 50 | Max labeled episodes per task in both held-out groups; 0 disables |
| `--no-holdout-cap` | off | Label held-out units in full instead of capping them at the task cap and the group budget |
| `--seed` | 0 | Sampling seed |

A held-out task must not appear in any other lab. Every scene it uses must also keep training episodes on other tasks.

## 3. Review

Each proposal lists checks:

- Violations, such as an ID episode whose group is missing from train, a held-out task that also appears in train, or a held-out lab that leaks into train. The summary table counts these.
- Notes: robots that exist only in the held-out labs, tasks the held-out labs share with train, and pinned held-out tasks with a near-duplicate train task. A near-duplicate is a pair with similarity ≥ 0.85, where similarity is the max of the difflib ratio and word-set Jaccard. These are not counted as violations but should be checked by hand.

Instructions are compared as strings, so paraphrases with low similarity still need a manual pass. Use the nearest-task column in the candidate table for this.

To look through a proposal's eval data, zip its three splits on LS6:

```bash
uv run oopsie splits package \
  --proposal /corral/utexas/IRI26024/oopsiedata/annotation/proposals/proposal_01.json \
  --data-dir /corral/utexas/IRI26024/oopsiedata
```

This writes `id.zip`, `od_task_id_scene.zip` and `od_task_od_scene.zip` into `/corral/utexas/IRI26024/oopsiedata/annotation/review_bundles/proposal_01/` by default. Each zip unpacks to `<split>/` in the layout of the annotation bundles (section 5), with episodes ordered by lab and task, so `oopsie-data annotate --samples-dir <split>/episodes` opens it. The copies are not blind: they keep their annotations and source data, and `eval_manifest.json` lists each episode's source, lab, robot and task. `--only SPLIT ...` packages a subset. The command exits non-zero if any episode lacks its h5 or videos.

## 3b. Sample train episodes for annotation (optional)

```bash
uv run python -m oopsie_backend.splits.sample_train \
  --index episode_index.parquet --proposal eval_split_proposals/proposal_02.json \
  --num-episodes 1000
```

This samples extra episodes to annotate from the proposal's train set. The pool excludes the proposal's eval and excluded episodes and every episode of a held-out lab or held-out task. The budget is split equally over labs, then within each lab over scenes, tasks, policies and success labels, so the sample maximizes coverage rather than following dataset proportions. A lab with fewer episodes than its share gives the rest to the other labs.

The output, `/corral/utexas/IRI26024/oopsiedata/annotation/train_annotation_sample.json` by default, uses the proposal layout with a single `train_annotated` split. Run steps 4 to 6 on it with the train directories, e.g. `--copies 1 --out-dir /corral/utexas/IRI26024/oopsiedata/annotation/train_assignment`.

In the [pipeline](pipeline.md), the `eval_proposals` stage writes the proposals to `oopsiedata/annotation/proposals/`. It passes `propose_args` from `configs/eval_split.yaml` to the script; these currently hold out tasks only (1000 `id` and 1000 `od_task_id_scene` episodes, at most 20 per task, from 58 tasks with up to 5 per lab). Setting `chosen_proposal` in `configs/eval_split.yaml` freezes that proposal as `annotation/eval_proposal.json`, records it in the index, samples the train episodes, and, once `annotation/annotators.txt` exists, writes the assignments. The commands below run the same steps by hand.

To record a proposal in the index by hand:

```bash
uv run python -m oopsie_backend.splits.record \
  --proposal eval_split_proposals/proposal_02.json --index episode_index.parquet
```

## 4. Assign annotators

```bash
uv run python -m oopsie_backend.annotation.assign \
  --proposal eval_split_proposals/proposal_01.json --annotators-file annotators.txt \
  --copies 2 --split-copies od_task_od_scene=3
```

Each eval episode goes to `--copies` distinct annotators. `--split-copies SPLIT=N` overrides this per split and can be repeated. `--annotators-file` reads one name per line and skips blank lines. Names must be unique and usable as file names. Alternatively, pass names directly with `--annotators NAME ...`, or use `--num-annotators N` for placeholder names.

Copies are assigned one at a time. Each copy goes to the annotator with the fewest samples of that split, lab and task, among annotators whose load is within one of the lowest. Ties go to the annotator who shares the fewest samples with the annotators already chosen for that episode, which spreads overlap evenly across annotator pairs. Each list is then shuffled so the same lab rarely appears twice in a row. The same seed and proposal give the same assignment.

Output, by default in `/corral/utexas/IRI26024/oopsiedata/annotation/`, next to the mirror's `OopsieData-Submissions/` folder, so every run sees it:

```
eval_assignment/
  annotator_01.json ...      # ordered episodes with position, split, lab, task, co_annotators
  samples.csv                # episode -> annotators
  assignment_summary.json    # loads, per-split and per-lab min/max across annotators, pair overlap
```

## 5. Package bundles (LS6)

```bash
uv run python -m oopsie_backend.annotation.package \
  --assignment-dir /corral/utexas/IRI26024/oopsiedata/annotation/eval_assignment \
  --data-dir /corral/utexas/IRI26024/oopsiedata
```

This writes one `<annotator>.zip` per annotator list into `/corral/utexas/IRI26024/oopsiedata/annotation/eval_bundles/` by default. Each zip unpacks to:

```
<annotator>/
  eval_manifest.json      # annotator, position, uid, bundle paths
  episodes/0001.h5
  episodes/0001_<cam>.mp4
  ...
```

Episodes are named by position because the annotation UI sorts by path, so it shows them in assignment order. Annotators run:

```bash
oopsie-data annotate --samples-dir <annotator>/episodes --annotator-name <name>
```

Copies are blind by default:

- `episode_annotations`, `source_data`, `source_*` attributes and `previous_episode_id` are removed.
- `policy_name` in `robot_profile` is set to `redacted`.
- `episode_id` is replaced by a stable hash of the source episode, so the same episode has the same id in every bundle.

- The manifest omits source paths and the proposal id. Collection maps the hashed ids back through the assignment directory.

`--keep-annotations` disables all of this. `--only NAME ...` packages a subset, and `--workers` sets how many bundles are written in parallel. The script exits non-zero if any episode lacks its h5 or videos.

### Distribute over Hugging Face

`annotation_pipeline` runs distribution and collection through one private dataset repo. It reads `configs/annotation_pipeline.yaml`, which names the repo, the data directory, the local staging and returns directories, and per batch the assignment directory, annotation store and release directory. Batches are named `batch_1`, `batch_2`, ... so annotators cannot tell which one is eval.

```bash
P="uv run python -m oopsie_backend.annotation.pipeline --config configs/annotation_pipeline.yaml"
$P stage      # LS6: package every batch into stage_dir
$P publish    # create the private batch and returns repos if needed and upload stage_dir
$P pull       # download annotated h5s from the returns repo's pull requests
$P collect    # add them to each batch's store
$P status     # returned/assigned per annotator and batch
$P release    # LS6: write source episodes with collected annotations
```

`stage` writes `<folder>/<batch>/` per annotator, where `<folder>` is the annotator name in lowercase ASCII with `_` separators, and a `README.md` with the annotator-to-folder table and the commands annotators run:

```bash
oopsie-data annotation download
oopsie-data annotate --samples-dir oopsie-annotation/<folder>/batch_1/episodes --annotator-name "<name>"
oopsie-data annotation upload
```

`annotation download` asks for the annotator's full name and a Hugging Face token and downloads only that annotator's folder of the batch repo (`repo_id`). `annotation upload` opens a pull request with every h5 annotated under that name on the returns repo (`returns_repo_id`), at the same paths. `publish` creates both repos as private datasets and refuses to upload if either exists and is public.

`pull` checks every open or merged pull request of the returns repo. A pull request is rejected if it changes anything other than `<folder>/<batch>/episodes/NNNN.h5` files of one known annotator folder. Accepted ones are downloaded to `returns_dir/pr_NNNN/`, with only the changed h5 files and, from the batch repo, the manifests of the touched batches. A pull request is downloaded again only when its head commit changes. `--close` closes open pull requests once downloaded. Pull requests never need to be merged, and `returns_dir/pull_requests.json` records the result for each.

`collect` passes every `pr_*/<folder>/<batch>` folder to `collect_eval_annotations` with that batch's assignment directory and store. Because the store is append-only and keyed by content, `pull` and `collect` can be re-run as returns arrive. When an annotator uploads twice, both versions are kept as revisions and the later timestamp is current.

## 6. Collect annotations

Annotators send back their bundle folder or a zip of it. The tool writes annotations into the h5 files.

```bash
uv run python -m oopsie_backend.annotation.collect_returns \
  returned/*.zip returned/annotator_07/ \
  --assignment-dir eval_assignment --store eval_annotations.jsonl
```

Each returned h5 is matched to its source episode by its blinded `episode_id`, so renamed files still match. Returns without `eval_manifest.json` need `--bundle NAME` to say whose list they are.

Every `episode_annotations/<name>` group becomes one record in the store, keyed by episode, bundle, name and content hash. The store is append-only:

- Re-collecting the same return adds nothing.
- A changed annotation is added as a new revision; the latest by `timestamp` is current.
- Annotations from different bundles never replace each other, even when two annotators typed the same name.

Each run also writes `eval_annotations_summary.json`. It reports returned vs assigned annotations, missing annotations per bundle, returns from annotators an episode was not assigned to, revisions, and outcome agreement per split.

To build the eval files, run:

```bash
uv run python -m oopsie_backend.annotation.collect_returns \
  --assignment-dir eval_assignment --store eval_annotations.jsonl \
  --apply-to eval_release --data-dir /corral/utexas/IRI26024/oopsiedata
```

This copies each annotated source episode, with its videos, into `eval_release/<repo>/`. It then adds every current annotation as its own `episode_annotations` group:

- A group is named after the annotator. If several bundles used the same name, it is named `<name>__<bundle>`.
- Groups written here carry `eval_bundle` and `eval_name` attributes, so a later revision rewrites only its own group.
- The lab's original annotations and all other groups are never modified.
- Re-running is safe.
