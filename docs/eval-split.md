# Eval split proposals

The eval set holds out three groups of episodes:

| Split | Definition | How it is chosen |
|---|---|---|
| `id` | Scene and task both present in train | Stratified sample over `(lab, robot, cameras, task)` groups |
| `od_task_id_scene` | Novel task, scene present in train | Full tasks from several medium multi-task labs, one task per lab |
| `od_task_od_scene` | Novel lab | Full small labs, on common and on custom robots |

A scene is approximated by `(lab, robot, cameras)`, because the h5 files have no background field. A task is the normalized language instruction: lowercased, with whitespace collapsed and surrounding punctuation stripped.

## 1. Build the index (LS6)

```bash
uv run python -m oopsie_backend.scripts.build_episode_index \
  --data-dir /corral/utexas/IRI26024/oopsiedata --out episode_index.jsonl --workers 16
```

This reads h5 metadata only and writes one JSON line per publicly included episode. Each line holds the lab, robot, cameras, instruction, policy, operator, duration, primary success label and annotators. The index is small, so you can copy it to a laptop and run the next step there.

## 2. Generate proposals

```bash
uv run python -m oopsie_backend.scripts.propose_eval_split \
  --index episode_index.jsonl --out-dir eval_split_proposals
```

Output:

```
eval_split_proposals/
  REPORT.md          # lab table, candidates, one section per proposal
  candidates.json
  proposal_01.json   # config, stats, checks, episode lists per split
  ...
```

An annotation budget, by default 2000 episodes, is split across the three groups by `--split-weights`. Each held-out lab and each held-out task gets an equal part of its group's budget as a target size. Candidates are ranked by `|log(size / target)|`. Once a unit is chosen, episodes beyond its target are moved to `excluded`, which is neither train nor eval.

"Medium" and "small" labs are defined by rank in the lab size ordering, not by share of the data. Task candidates must have no near-duplicate anywhere in the dataset. The candidate table lists each lab's best task before any lab's second, so its top rows span several labs.

Without pins, the script crosses the top `--candidates` (default 3) sets of each holdout type, which gives up to 27 proposals. Repeated pins of one kind form a single set that replaces the search for that kind. For example, `--od-task 'A::x' --od-task 'B::y'` holds out both tasks.

| Flag | Default | Meaning |
|---|---|---|
| `--eval-budget` / `--budget-frac` | 2000 / 0.05 | Episodes to label in total; pass `--eval-budget -1` to use the fraction instead |
| `--split-weights` | 0.2,0.4,0.4 | Budget shares of `id`, `od_task_id_scene` and `od_task_od_scene` |
| `--num-od-tasks` | 3 | Held-out tasks, each from a different lab |
| `--num-ood-common` / `--num-ood-custom` | 1 / 1 | Held-out labs per embodiment class |
| `--ood-common LAB` / `--ood-custom LAB` | search | Pin held-out labs |
| `--od-task 'LAB::instruction'` | search | Pin held-out tasks |
| `--common-robot` / `--custom-robot` | auto | Override robot class. Auto means common if at least two labs use it, after lowercasing and replacing punctuation |
| `--ood-lab-min` | 40 | Minimum size of a held-out lab |
| `--ood-lab-max-quantile` | 0.5 | Held-out labs come from the smaller half of labs |
| `--od-task-lab-quantiles` | 0.25,0.75 | Size-rank window of labs that may lose a task |
| `--od-task-min-tasks` / `--od-task-min-episodes` | 3 / 25 | Such a lab needs this many tasks with at least this many episodes |
| `--od-task-max-share` | 0.4 | A held-out task may hold at most this share of its lab |
| `--id-alpha` | 0.5 | ID group weight is `size**alpha`; 1 is proportional, 0 is uniform |
| `--id-max-group-frac` | 0.25 | Max fraction of a group taken for ID eval |
| `--id-min-group` | 8 | Groups smaller than this are never sampled |
| `--no-holdout-cap` | off | Label held-out units in full instead of capping them at their target |
| `--seed` | 0 | Sampling seed |

A held-out task must not appear in any other lab. Every scene it uses must also keep training episodes on other tasks.

## 3. Review

Each proposal lists checks:

- Violations, such as an ID episode whose group is missing from train, a held-out task that also appears in train, or a held-out lab that leaks into train. The summary table counts these.
- Notes: robots that exist only in the held-out labs, tasks the held-out labs share with train, and pinned held-out tasks with a near-duplicate train task. A near-duplicate is a pair with similarity ≥ 0.85, where similarity is the max of the difflib ratio and word-set Jaccard. These are not counted as violations but should be checked by hand.

Instructions are compared as strings, so paraphrases with low similarity still need a manual pass. Use the nearest-task column in the candidate table for this.

## 3b. Sample train episodes for annotation (optional)

```bash
uv run python -m oopsie_backend.scripts.sample_train_annotations \
  --index episode_index.jsonl --proposal eval_split_proposals/proposal_02.json \
  --num-episodes 1000 --out train_annotation_sample.json
```

This samples extra episodes to annotate from the proposal's train set. The pool excludes the proposal's eval and excluded episodes and every episode of a held-out lab or held-out task. The budget is split equally over labs, then within each lab over scenes, tasks, policies and success labels, so the sample maximizes coverage rather than following dataset proportions. A lab with fewer episodes than its share gives the rest to the other labs.

The output uses the proposal layout with a single `train_annotated` split. Run steps 4 to 6 on it with separate directories, e.g. `--copies 1 --out-dir train_assignment`, then `train_bundles`, `train_annotations.jsonl` and `train_release`.

## 4. Assign annotators

```bash
uv run python -m oopsie_backend.scripts.assign_eval_annotators \
  --proposal eval_split_proposals/proposal_01.json --annotators-file annotators.txt \
  --copies 2 --split-copies od_task_od_scene=3 --out-dir eval_assignment
```

Each eval episode goes to `--copies` distinct annotators. `--split-copies SPLIT=N` overrides this per split and can be repeated. `--annotators-file` reads one name per line and skips blank lines. Names must be unique and usable as file names. Alternatively, pass names directly with `--annotators NAME ...`, or use `--num-annotators N` for placeholder names.

Copies are assigned one at a time. Each copy goes to the annotator with the fewest samples of that split, lab and task, among annotators whose load is within one of the lowest. Ties go to the annotator who shares the fewest samples with the annotators already chosen for that episode, which spreads overlap evenly across annotator pairs. Each list is then shuffled so the same lab rarely appears twice in a row. The same seed and proposal give the same assignment.

Output:

```
eval_assignment/
  annotator_01.json ...      # ordered episodes with position, split, lab, task, co_annotators
  samples.csv                # episode -> annotators
  assignment_summary.json    # loads, per-split and per-lab min/max across annotators, pair overlap
```

## 5. Package bundles (LS6)

```bash
uv run python -m oopsie_backend.scripts.package_eval_bundles \
  --assignment-dir eval_assignment \
  --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir eval_bundles
```

This writes one `<annotator>.zip` per annotator list. Each zip unpacks to:

```
<annotator>/
  eval_manifest.json      # position, uid, bundle paths, source dataset_repo + repo_path
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

`--keep-annotations` disables all of this. `--only NAME ...` packages a subset, and `--workers` sets how many bundles are written in parallel. The script exits non-zero if any episode lacks its h5 or videos.

## 6. Collect annotations

Annotators send back their bundle folder or a zip of it. The tool writes annotations into the h5 files.

```bash
uv run python -m oopsie_backend.scripts.collect_eval_annotations \
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
uv run python -m oopsie_backend.scripts.collect_eval_annotations \
  --assignment-dir eval_assignment --store eval_annotations.jsonl \
  --apply-to eval_release --data-dir /corral/utexas/IRI26024/oopsiedata
```

This copies each annotated source episode, with its videos, into `eval_release/<repo>/`. It then adds every current annotation as its own `episode_annotations` group:

- A group is named after the annotator. If several bundles used the same name, it is named `<name>__<bundle>`.
- Groups written here carry `eval_bundle` and `eval_name` attributes, so a later revision rewrites only its own group.
- The lab's original annotations and all other groups are never modified.
- Re-running is safe.
