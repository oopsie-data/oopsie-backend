# Spot checks

Spot checks are a manual review of a small sample per lab. They catch what the automated checks cannot judge: wrong instructions, mislabeled outcomes, odd camera placements. The commands select the episodes, package them into zip bundles for reviewers, and merge the returned notes into the index. They are not part of the pipeline.

## Selecting episodes

Preview the selection first; this writes nothing:

```bash
uv run oopsie qa select-spot-checks --data-dir /corral/utexas/IRI26024/oopsiedata --dry-run
```

It prints one row per lab with the number of selected and available episodes and how many distinct instructions, annotators, operators and robots the selection covers. The selection spreads over these values, so that a reviewer sees as much of each lab's variety as possible. Labs excluded from the public stats are never used, and episodes without readable videos are skipped.

Then write the bundles:

```bash
uv run oopsie qa select-spot-checks --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir qa_out
```

| Option | Default | Effect |
|---|---|---|
| `--per-lab` | 20 | episodes per lab |
| `--min-episodes` | 200 | skip labs with fewer eligible episodes; `0` for test data |
| `--labs-per-bundle` | 5 | labs per zip, so 100 episodes by default |
| `--seed` | 0 | the same seed and data give the same bundles |
| `--workers` | 8 | parallel h5 readers |

The output holds one zip per bundle and `qa_master_manifest.json`, which lists every bundle's episodes; keep it. Each zip unpacks to `qa_bundle_XX/qa_manifest.json` and `episodes/<repo>/<repo_path>` with the h5 and video files. A manifest entry identifies the episode (`dataset_repo`, `repo_path`, `episode_id`), describes it (lab, instruction, operator, annotators, robot) and has an empty `qa_notes` field for the reviewer.

## Reviewing

Send each reviewer one zip and the reviewer instructions, `QA_REVIEW.md` in the `qa_tool` branch of `oopsie-data-tools`, and keep track of who has which bundle. Reviewers return their `qa_manifest.json` with notes filled in.

## Merging notes

Save the returned manifests under distinct names, then merge them into a table or into the index:

```bash
uv run oopsie qa merge-notes returned/*.json --out qa_notes.csv      # add --all to include episodes without notes
uv run oopsie qa merge-notes returned/*.json --index episode_index.parquet
```

The index variant writes the `qa_notes` column: bundle, QA id, reviewer, time and text of each note.
