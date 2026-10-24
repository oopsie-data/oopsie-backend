# QA spot checks: generating bundles

Run from the `oopsie-backend` repo root, on a machine with the synced data mirror (LS6: `/corral/utexas/IRI26024/oopsiedata`).

## 1. Preview the selection

```bash
uv run python -m oopsie_backend.qa.select_spot_checks \
  --data-dir /corral/utexas/IRI26024/oopsiedata --dry-run
```

This prints one row per selected lab, showing the selected/available episode count and the distinct values selected/available for instruction, annotator, operator and robot. Skipped labs and unreadable episodes are logged to stderr. Nothing is written.

## 2. Write the bundles

```bash
uv run python -m oopsie_backend.qa.select_spot_checks \
  --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir qa_out
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `/corral/utexas/IRI26024/oopsiedata` | Root of the synced mirror (contains `OopsieData-Submissions/`) |
| `--out-dir` | `qa_out` | Output directory |
| `--per-lab` | `20` | Episodes per lab |
| `--min-episodes` | `200` | Skip labs with fewer eligible episodes (`0` for test data) |
| `--labs-per-bundle` | `5` | Labs per zip (5 × 20 = 100 episodes) |
| `--seed` | `0` | Selection seed; same seed and data give the same bundles |
| `--workers` | `8` | Parallel h5 readers |
| `--dry-run` | off | Print the selection table only |

Labs are HF dataset repos. Repos excluded from the public stats are never used. Episodes without readable videos are skipped.

Output:

```
qa_out/
  qa_bundle_01.zip, qa_bundle_02.zip, ...
  qa_master_manifest.json      # every bundle's manifest, keep this
```

Each zip unpacks to `qa_bundle_XX/qa_manifest.json` plus `episodes/<Repo>/<repo_path>` (h5 and mp4s). Each manifest entry has `qa_id`, `dataset_repo`, `repo_path`, `bundle_path`, `video_files`, `lab_id`, `episode_id`, `language_instruction`, `operator`, `annotators`, `robot_name` and `qa_notes`.

## 3. Hand out

Send each reviewer one zip and the reviewer instructions (`QA_REVIEW.md` in the `qa_tool` branch of oopsie-data-tools). Track which zip went to whom.

## 4. Collect the returned notes

Reviewers return their `qa_manifest.json`. Save the files under distinct names, then:

```bash
uv run python -m oopsie_backend.qa.merge_notes returned/*.json --out qa_notes.csv
```

`--all` also includes episodes without notes. Each row identifies the episode by `dataset_repo` + `repo_path` (the h5 path inside that HF repo), together with the bundle, reviewer and note.

## Local test run

```bash
HF_TOKEN=hf_... uv run python -m oopsie_backend.ingest.sync \
  --org OopsieData-Submissions --local-dir ./data --smallest 10
uv run python -m oopsie_backend.qa.select_spot_checks \
  --data-dir ./data --out-dir qa_out --min-episodes 0
```
