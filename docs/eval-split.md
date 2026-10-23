# Eval split proposals

The eval set holds out three groups of episodes:

| Split | Definition | How it is chosen |
|---|---|---|
| `id` | Scene and task both present in train | Stratified sample over `(lab, robot, cameras, task)` groups |
| `od_task_id_scene` | Novel task, scene present in train | One full task from a medium lab with several tasks |
| `od_task_od_scene` | Novel lab | Two full small labs: one on a common robot, one on a custom robot |

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

Without pins, the script crosses the top `--candidates` (default 3) of each holdout type, which gives up to 27 proposals. Repeated pins are alternatives, not unions: `--ood-common A --ood-common B` yields proposals with A and proposals with B.

| Flag | Default | Meaning |
|---|---|---|
| `--ood-common LAB` / `--ood-custom LAB` | search | Pin the held-out lab per embodiment class |
| `--od-task 'LAB::instruction'` | search | Pin the held-out task |
| `--common-robot` / `--custom-robot` | auto | Override robot class; auto = common if used by ≥2 labs |
| `--id-frac` | 0.05 | ID eval size as a fraction of all episodes |
| `--id-alpha` | 0.5 | Group weight `size**alpha`; 1 is proportional, 0 is uniform |
| `--id-max-group-frac` | 0.25 | Max fraction of a group taken for ID eval |
| `--id-min-group` | 8 | Groups smaller than this are never sampled |
| `--ood-lab-min` / `--ood-lab-max-frac` | 40 / 0.03 | Size window for held-out labs |
| `--od-task-lab-min-frac` / `--od-task-lab-max-frac` | 0.02 / 0.15 | Size window for the lab that loses a task |
| `--od-task-min-tasks` / `--od-task-min-episodes` | 3 / 25 | The lab needs this many tasks with at least this many episodes |
| `--od-task-max-share` | 0.4 | A held-out task may hold at most this share of its lab |
| `--holdout-cap` | none | Max eval episodes per held-out lab or task. Episodes over the cap go to `excluded`, which is neither train nor eval |
| `--seed` | 0 | Sampling seed |

A held-out task candidate must not appear in any other lab. Every scene it uses must also keep training episodes on other tasks.

## 3. Review

Each proposal lists checks:

- Violations, such as an ID episode whose group is missing from train, a held-out task that also appears in train, or a held-out lab that leaks into train. The summary table counts these.
- Notes: robots that exist only in the held-out labs, tasks the held-out labs share with train, and held-out tasks with a near-duplicate train task. A near-duplicate is a pair with similarity ≥ 0.85, where similarity is the max of the difflib ratio and word-set Jaccard. These are not counted as violations but should be checked by hand.

Instructions are compared as strings, so paraphrases with low similarity still need a manual pass. Use the nearest-task column in the candidate table for this.
