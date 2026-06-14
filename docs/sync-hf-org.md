# sync_hf_org

Sync every dataset under a Hugging Face organization to a local directory. Downloads only missing or changed files (ETag-based). Files removed from the Hub are **kept locally** and listed as orphans in the sync report — local storage is a superset of the Hub, not a strict mirror.

## Command

```bash
cd /work/09666/abahety/ls6/oopsie-backend

export HF_TOKEN=hf_...   # required for private org datasets

uv run python -m oopsie_backend.scripts.sync_hf_org \
  --org OopsieData-Submissions \
  --local-dir /scratch/09666/abahety/oopsiedata
```

Org can also be a profile URL:

```bash
uv run python -m oopsie_backend.scripts.sync_hf_org \
  --org https://huggingface.co/OopsieData-Submissions \
  --local-dir /scratch/09666/abahety/oopsiedata
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--org` | *(required)* | HF organization slug or profile URL |
| `--local-dir` | *(required)* | Root directory for synced local data |
| `--token` | `$HF_TOKEN` | HF read token |
| `--dry-run` | off | Report what would download without writing files |
| `--verbose` | off | Per-dataset timing and destination paths |
| `--limit-datasets` | none | Cap number of datasets (for testing) |

Full flag list: `uv run python -m oopsie_backend.scripts.sync_hf_org --help`

## Local layout

Each dataset repo is stored under:

```
<local-dir>/<namespace>/<repo-name>/
```

All file types are synced (`.h5`, `.mp4`, `.json`, etc.) — there is no extension filter.

## Expected stderr output

### Default (no `--verbose`)

```
  Syncing 18 dataset(s) under 'OopsieData-Submissions'
  [1/18] OopsieData-Submissions/SomeLab  downloaded 12, skipped 450, took 2m 15s
  [2/18] OopsieData-Submissions/OtherLab  downloaded 0, skipped 1203, took 45.2s
  ...

  Downloaded: 12 file(s), skipped (unchanged): 37041, datasets: 18, orphans kept: 847, errors: 0
  Report: /scratch/09666/abahety/oopsiedata/.sync_report.json
```

When already fully synced, `downloaded` is 0 and `skipped` is high.

### With `--verbose`

```
  Syncing 18 dataset(s) under 'OopsieData-Submissions'
  [1/18] OopsieData-Submissions/SomeLab
    dest: /scratch/09666/abahety/oopsiedata/OopsieData-Submissions/SomeLab
    downloaded 12, skipped 450, took 2m 15s

  Checking orphans...
  OopsieData-Submissions/SomeLab: 3 orphan(s)

  Downloaded: 12 file(s), skipped (unchanged): 37041, datasets: 18, orphans kept: 847, errors: 0
  Report: /scratch/09666/abahety/oopsiedata/.sync_report.json
```

### Dry run

```
  Dry-run 18 dataset(s) under 'OopsieData-Submissions'
  [1/18] OopsieData-Submissions/SomeLab  would download 5, skipped 457, took 12.3s
  ...

  Would download: 5 file(s), skipped (unchanged): 37048, datasets: 18, orphans kept: 0, errors: 0
  Report: /scratch/09666/abahety/oopsiedata/.sync_report.json
```

Orphan detection is skipped during dry runs.

## Sync report (`.sync_report.json`)

Written to `<local-dir>/.sync_report.json` after every run.

Example (truncated):

```json
{
  "org": "OopsieData-Submissions",
  "local_dir": "/scratch/09666/abahety/oopsiedata",
  "dry_run": false,
  "synced_at": {
    "utc": "2026-06-14T16:22:34Z",
    "cdt": "2026-06-14T11:22:34-05:00"
  },
  "datasets_synced": 18,
  "files_downloaded": 0,
  "files_skipped": 37053,
  "orphans_kept": [
    "OopsieData-Submissions/UT_RobIn/20260422_220912.h5",
    "OopsieData-Submissions/UT_RobIn/20260422_220912_left.mp4"
  ],
  "errors": [],
  "datasets": [
    "OopsieData-Submissions/SomeLab",
    "OopsieData-Submissions/OtherLab"
  ]
}
```

| Field | Meaning |
|-------|---------|
| `files_downloaded` | New or updated files written this run |
| `files_skipped` | Files already present and unchanged |
| `orphans_kept` | Local files no longer on the Hub (not deleted) |
| `errors` | Datasets that could not be synced (e.g. not found) |
| `datasets` | Repos successfully processed this run |

## Exit codes and errors

| Condition | Exit code | Message |
|-----------|-----------|---------|
| Success | 0 | Summary + report path printed |
| No datasets visible to token | 1 | `error: no datasets returned for author=...` |
| Repo not found for one dataset | 0 | Logged as `skip (not found): ...`, counted in `errors` |

## See also

- [Sync and stats](sync-and-stats.md)
- [Compute stats](compute-stats.md) — next step after sync
