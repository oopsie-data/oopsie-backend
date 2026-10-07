# QA spot checks

## 1. Select episodes and build bundles

```bash
uv run python -m oopsie_backend.scripts.select_qa_episodes \
  --data-dir /corral/utexas/IRI26024/oopsiedata --out-dir qa_out
```

- One lab is one HF dataset repo. Repos in `STATS_EXCLUDED_DATASETS` / `PUBLIC_EXCLUDED_DATASETS` are skipped, as are labs with fewer than `--min-episodes` (default 200) episodes that have readable videos.
- Per lab, `--per-lab` (default 20) episodes are picked greedily to cover as many distinct language instructions, annotators, operators and robots (`robot_profile.robot_name`) as possible. Values repeat only once a dimension is exhausted. Selection is deterministic for a given `--seed`.
- Labs are grouped `--labs-per-bundle` (default 5) per `qa_bundle_XX.zip`. `qa_master_manifest.json` lists every bundle.
- `--dry-run` prints, per lab, how many distinct values were selected out of those available, and writes nothing.

Bundle layout:

```
qa_bundle_01/
  qa_manifest.json
  episodes/<Repo>/<repo_path>.h5   (+ its mp4s at their repo-relative locations)
```

Each `qa_manifest.json` entry holds `qa_id`, `dataset_repo`, `repo_path`, `bundle_path`, `video_files`, `lab_id`, `episode_id`, `language_instruction`, `operator`, `annotators`, `robot_name` and an empty `qa_notes`.

## 2. Review (reviewer side)

Requires `oopsie-data-tools` with the `qa` command:

```bash
oopsie-data qa qa_bundle_01.zip --reviewer <name>
```

This unpacks the zip next to itself and opens the annotator in QA mode. Episodes and existing annotations are read-only, and the only input is a QA note per episode, saved into `qa_bundle_01/qa_manifest.json`. Re-running the command resumes. When done, the reviewer sends back `qa_manifest.json`.

## 3. Collect notes

```bash
uv run python -m oopsie_backend.scripts.merge_qa_notes returned/*.json --out qa_notes.csv
```

Each row identifies the episode by `dataset_repo` + `repo_path`. Pass `--all` to include episodes without notes.

## Local test data

```bash
HF_TOKEN=... uv run python -m oopsie_backend.scripts.sync_hf_org \
  --org OopsieData-Submissions --local-dir ./data --smallest 10
uv run python -m oopsie_backend.scripts.select_qa_episodes \
  --data-dir ./data --out-dir qa_out --min-episodes 0
```
