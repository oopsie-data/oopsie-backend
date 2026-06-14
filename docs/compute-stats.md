# Compute stats

Two scripts produce the same `stats.json` schema for the [`oopsie-docs`](../../oopsie-docs) dashboard. Use **local** stats in production (after sync); use **HF** stats for ad-hoc checks without syncing to disk first.

## Output location

Default path (sibling `oopsie-docs` repo):

```
/work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
```

Override with `--output` or the `OOPSIE_STATS_OUT` environment variable.

---

## compute_local_stats

Walk synced local HF data, read every `.h5` file, and write aggregated dashboard stats. This is what the daily Slurm job runs.

### Command

```bash
cd /work/09666/abahety/ls6/oopsie-backend

uv run python -m oopsie_backend.scripts.compute_local_stats \
  --data-dir /scratch/09666/abahety/oopsiedata
```

Explicit output path (optional — matches the default):

```bash
uv run python -m oopsie_backend.scripts.compute_local_stats \
  --data-dir /scratch/09666/abahety/oopsiedata \
  --output /work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `/scratch/09666/abahety/oopsiedata` | Root of synced local data |
| `--output` | `DEFAULT_STATS_JSON` | Where to write `stats.json` |
| `--limit` | none | Max `.h5` files per dataset (first N, for testing) |

No `HF_TOKEN` required — reads files from disk.

### Expected stderr output

```
Found 10675 .h5 file(s) under /scratch/09666/abahety/oopsiedata
  dataset OopsieData-Submissions/LabA: 1200 .h5 file(s)
  dataset OopsieData-Submissions/LabB: 450 .h5 file(s)
  ...

Wrote /work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
  10675 episodes, 37 labs, 9 datasets, 1191S / 8922F over 104.179h
```

Per-file read failures are logged and counted in `totals.read_errors`:

```
  [3/120] OopsieData-Submissions/LabA :: path/to/bad.h5: SKIPPED (...)
```

### Errors (exit 1)

```
error: data dir not found: /path/to/missing
error: no .h5 files found under /path/to/empty
```

`.h5` files must live at least three levels below `--data-dir` (`<namespace>/<repo>/<path>`). Files under `.cache/` are ignored.

---

## compute_hf_stats

Stream `.h5` files from Hugging Face into memory (nothing persisted except `stats.json`). Useful for quick checks without syncing the full org.

### Command

Single dataset:

```bash
export HF_TOKEN=hf_...

uv run python -m oopsie_backend.scripts.compute_hf_stats \
  --repo OopsieData-Submissions/SomeLab
```

Entire org (same scope as sync):

```bash
uv run python -m oopsie_backend.scripts.compute_hf_stats \
  --org OopsieData-Submissions
```

Dataset URL form:

```bash
uv run python -m oopsie_backend.scripts.compute_hf_stats \
  --repo https://huggingface.co/datasets/OopsieData-Submissions/SomeLab
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` / `--org` | *(one required)* | Single dataset or entire org |
| `--repo-type` | `dataset` | `dataset` or `model` |
| `--token` | `$HF_TOKEN` | HF read token |
| `--output` | `DEFAULT_STATS_JSON` | Where to write `stats.json` |
| `--limit` | none | Max `.h5` files per dataset (for testing) |

### Expected output

Single repo:

```
Found 450 .h5 file(s) in OopsieData-Submissions/SomeLab (dataset)

Wrote /work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
  450 episodes, 1 labs, 120S / 330F over 12.5h
```

Org sweep:

```
Found 18 dataset repo(s) under org 'OopsieData-Submissions'
  dataset OopsieData-Submissions/LabA: 1200 .h5 file(s)
  skip (no .h5): OopsieData-Submissions/EmptyRepo
  ...

Wrote /work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json
  10675 episodes, 37 labs, 1191S / 8922F over 104.179h
```

Note: org-mode summary line omits `datasets` count (unlike `compute_local_stats`).

### Errors (exit 1)

```
error: repository not found: 'namespace/repo' (repo_type='dataset').
error: no datasets returned for author='SomeOrg'.
error: could not list datasets for org 'SomeOrg': ...
```

---

## stats.json schema

Both scripts write `schema_version: 2` JSON. Top-level fields:

| Field | Description |
|-------|-------------|
| `generated_at` | `{utc, cdt}` timestamps |
| `dataset_repo` | Source label (`local:/path`, repo id, or `org/*`) |
| `totals` | Episode/lab/robot counts, successes, failures, `read_errors` |
| `per_lab`, `per_dataset`, `per_robot`, `per_policy` | Breakdowns |
| `top_tasks`, `failure_categories`, `severity_distribution` | Aggregates |
| `submission_timeline` | Episodes per day |
| `datasets_included` | List of dataset repos processed (local + org HF modes) |

Example `totals` block:

```json
{
  "episodes": 10675,
  "labs": 37,
  "datasets": 9,
  "robots": 7,
  "policies": 22,
  "tasks": 2580,
  "duration_hours": 104.179,
  "annotated_episodes": 10113,
  "successes": 1191,
  "failures": 8922,
  "failure_descriptions": 267,
  "read_errors": 0
}
```

The dashboard reads this file from `oopsie-docs/docs/assets/data/stats.json`.

## See also

- [Sync and stats](sync-and-stats.md)
- [sync_hf_org](sync-hf-org.md) — sync HF data to disk first
