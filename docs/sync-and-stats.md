# Sync and stats

The LS6 daily job syncs Hugging Face org datasets to local scratch, then regenerates the dashboard `stats.json` consumed by the sibling [`oopsie-docs`](../../oopsie-docs) Jekyll site.

Local storage is a **superset** of the Hub: new and changed files are downloaded, but files removed from the Hub are kept on disk (reported as orphans). This is sync, not a strict mirror.

```mermaid
flowchart LR
  HF["HuggingFace org"] -->|sync_hf_org| Local["Local copy\n/scratch/.../oopsiedata"]
  Local -->|compute_local_stats| Stats["stats.json\noopsie-docs/docs/assets/data/"]
  Local -->|compute_local_stats| Reports["reports/\npublic + internal JSON/HTML"]
  Stats --> Dashboard["oopsie-docs dashboard"]
```

## Prerequisites

From the `oopsie-backend` directory:

```bash
cd /work/09666/abahety/ls6/oopsie-backend
uv sync
```

Set a Hugging Face read token and correct file permissions:

```bash
export HF_TOKEN=hf_...
# or
echo hf_xxxxxxxxxxxxxxxx > ~/.hf_token
chmod 600 ~/.hf_token

# This is very inmportant!
echo 'umask 007' >> ~/.bashrc
```

On LS6 `vm-small` nodes, numpy must stay below 2.4 (older CPUs lack x86-v2). The daily job handles this automatically; for manual runs:

```bash
uv add "numpy<2.4" && uv sync --reinstall-package numpy
```

## (Option 1) Daily Slurm job

Batch scripts live in [`slurm/`](../slurm/):

- [`daily_oopsie.slurm`](../slurm/daily_oopsie.slurm) — sync + stats
- [`daily.scron`](../slurm/daily.scron) — cron schedule (3:00 daily)

Submit from anywhere:

```bash
sbatch /work/09666/abahety/ls6/oopsie-backend/slurm/daily_oopsie.slurm
```

if vm-small is taking too long, try:

```bash
# check which nodes are free
sinfo -S+P -o "%18P %10l %20F"

# Use them
sbatch -p normal /work/09666/abahety/ls6/oopsie-backend/slurm/daily_oopsie.slurm

# Can also reduce time for node which gives you quicker access
sbatch -p normal,vm-small -t 06:00:00 /work/09666/abahety/ls6/oopsie-backend/slurm/daily_oopsie.slurm
```

Expected log structure (stdout and stderr merged into one file):

```
=== 2026-06-14T10:00:00-05:00 START: setup_numpy ===
...
=== 2026-06-14T10:00:05-05:00 SUCCESS: setup_numpy completed in 5s ===
=== 2026-06-14T10:00:05-05:00 START: sync_hf_org ===
  Syncing 18 dataset(s) under 'OopsieData-Submissions'
  ...
=== 2026-06-14T10:15:00-05:00 SUCCESS: sync_hf_org completed in 895s ===
=== 2026-06-14T10:15:00-05:00 START: compute_local_stats ===
  ...
=== 2026-06-14T10:20:00-05:00 SUCCESS: compute_local_stats completed in 300s ===
=== 2026-06-14T10:20:00-05:00 SUCCESS: oopsie-daily completed in 1200s (job 12345) ===
```
Two useful outputs:
1. Logs: `/corral/utexas/IRI26024/logs/oopsie-daily-<jobid>.log`
2. Final report: `/corral/utexas/IRI26024/oopsiedata/.sync_report.json`

## (Option 2) Manual run (two steps)

### 1. Sync

```bash
cd /work/09666/abahety/ls6/oopsie-backend

export HF_TOKEN=hf_...   # required for private org datasets
export HF_HUB_DISABLE_XET=1   # this avoids reaching HUB API limits!

uv run python -m oopsie_backend.scripts.sync_hf_org \
  --org OopsieData-Submissions \
  --local-dir /corral/utexas/IRI26024/oopsiedata \
  --group G-829069 \
  --verbose
```
Note: Currently we are using snapshot_download, might have to switch to git clone/git pull in order to avoid hitting the API limits and making the script run faster

#### Permissions

Synced data must be accessible to all members of group `G-829069`. The sync runs with umask `002`, and with `--group G-829069` it additionally assigns every path under `--local-dir` it owns to that group. Files get group read/write, and directories get group `rwx` plus setgid, so new files inherit the group. Paths owned by other users are skipped. Those paths need a one-off `chgrp -R G-829069` by their owner. The daily Slurm job passes `--group G-829069` and also sets the group on `$DATA_DIR` before syncing.

To fix existing data by hand, the owner of the files runs:

```bash
DATA_DIR=/corral/utexas/IRI26024/oopsiedata
chgrp -R G-829069 "$DATA_DIR"
chmod -R g+rwX "$DATA_DIR"
find "$DATA_DIR" -type d -exec chmod g+s {} +
```

`g+rwX` gives the group execute permission only on directories and on files that are already executable. To find paths that still have the wrong group, run `find "$DATA_DIR" ! -group G-829069`.

#### Expected stderr output

##### Default (no `--verbose`)

```
  Syncing 18 dataset(s) under 'OopsieData-Submissions'
  [1/18] OopsieData-Submissions/SomeLab  downloaded 12, skipped 450, took 2m 15s
  [2/18] OopsieData-Submissions/OtherLab  downloaded 0, skipped 1203, took 45.2s
  ...

  Downloaded: 12 file(s), skipped (unchanged): 37041, datasets: 18, orphans kept: 847, errors: 0
  Report: /scratch/09666/abahety/oopsiedata/.sync_report.json
```

When already fully synced, `downloaded` is 0 and `skipped` is high.

##### With `--verbose`

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

##### Sync report (`.sync_report.json`)

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


### 2. Update stats

```bash
uv run python -m oopsie_backend.scripts.compute_local_stats \
  --data-dir /corral/utexas/IRI26024/oopsiedata
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--no-public-report` | off | Skip `public_stats.html` |
| `--no-internal-output` | off | Skip internal JSON/HTML entirely |
| `--no-internal-report` | off | Skip `internal_stats.html` only |
| `--limit` | none | Max `.h5` files per dataset (first N, for testing) |

No `HF_TOKEN` required — reads files from disk.

Internal stats additionally make a **best-effort** call to the Hub (using a cached
`huggingface-cli login` token or `HF_TOKEN`, if either is set) to list every dataset
repo in the org and its true creation date, for the "All datasets" inventory in
`internal_stats.html` (sortable by created/episodes/hours; includes repos with zero
episodes, e.g. labs that registered but haven't submitted yet). If no token is
available or the Hub can't be reached, this is skipped silently and that inventory
falls back to each dataset's earliest-episode timestamp instead.

### Config: `configs/stats_config.yaml`

All of the following are edited there, not in code - changes take effect on the
next `compute_local_stats` / `compute_hf_stats` run:

- The four dataset/lab exclusion lists (archives, bulk-import, non-lab infra repos,
  placeholder lab ids) - these used to be hardcoded constants in `episode_stats.py`.
- The list of placeholder operator names (e.g. `pending_operator`, `unknown`) that
  never count as a real individual.
- The alpha/beta lab requirement thresholds and on/off toggles that drive the
  colored status dot on each row of the "All datasets" inventory: a lab is
  "alpha" if any of its dataset repos predates `alpha_cutoff_date`, otherwise
  "beta"; a lab *passes* its tier if even one individual clears every *enabled*
  clause for that tier (never summed across people). `task_diversity` and
  `per_task_cap` are placeholders - not implemented yet, always disabled.

If the file is missing, the code falls back to built-in defaults matching the
shipped YAML, so this never hard-fails - but you should expect to edit
`configs/stats_config.yaml`, not `episode_stats.py`, going forward.

`compute_local_stats` writes two things: 
1. public and internal stats to `reports/` 
2. the public stats json to oopsie-docs.


## Paths (LS6)

| Variable / path | Purpose |
|-----------------|---------|
| `/corral/utexas/IRI26024/oopsiedata` | Root of synced local data |
| `/corral/utexas/IRI26024/oopsiedata/.sync_report.json` | Last sync summary |
| `/work/09666/abahety/ls6/oopsie-docs/docs/assets/data/stats.json` | Public dashboard stats (docs site) |
| `/work/09666/abahety/ls6/oopsie-backend/reports/public_stats.json` | Public stats copy (backend) |
| `/work/09666/abahety/ls6/oopsie-backend/reports/public_stats.html` | Self-contained public stats snapshot |
| `/work/09666/abahety/ls6/oopsie-backend/reports/internal_stats.json` | Full internal stats (all repos; not published) |
| `/work/09666/abahety/ls6/oopsie-backend/oopsie-backend/reports/internal_stats.html` | Self-contained internal report |

