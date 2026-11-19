# LS6 runbook

The pipeline runs on TACC's Lonestar6 cluster under allocation `IRI23005`. A nightly job brings everything up to date, and manual targets such as training or imports are submitted as separate jobs. This page covers how the jobs are set up, how to check on them and what to do when they fail.

## Storage

`/corral/utexas/IRI26024` holds data only. TACC does not intend Corral for computation and limits the number of files per allocated terabyte, so jobs keep their code, tools and caches on `$SCRATCH` and copy the episodes they read heavily to node-local disk.

| Path | Contents |
|---|---|
| `/corral/utexas/IRI26024/oopsiedata/` | the mirror; `OopsieData-Submissions/`, `imports/`, `annotation/`, `.sync_state.parquet`, `.sync_report.json`, `tokens.yaml` |
| `/corral/utexas/IRI26024/episode_index/` | the index and the pipeline cache |
| `/corral/utexas/IRI26024/reports/` | QA, stats and analysis reports |
| `/corral/utexas/IRI26024/training/` | materialized training sets, one folder per recipe |
| `/corral/utexas/IRI26024/logs/` | job logs |
| `$SCRATCH/oopsie/` | the staged backend checkout, uv and its Python, the uv and Hugging Face caches, imports being converted, training sets being built |
| `/tmp/oopsie/episodes` on each node | episodes that QA and training are reading, at most 1,000 at a time |

The index and the cache stay on Corral, because every job that writes the index must use the same copy. A purge of `$SCRATCH` only means that the next job downloads uv and copies the caches again. The canonical copy of the code is the git remote.

## Permissions

All files must belong to the project group `G-829069`. Corral accounts quota by group, so a file of another group fails with `Disk quota exceeded`. The directories under `/corral/utexas/IRI26024` have the setgid bit, so new files inherit the group, and the jobs reset the group and modes of the index, reports and logs at the end. When running commands by hand, use `sg G-829069 -c "<command>"`, which makes the group the default even where setgid is missing.

To repair files that ended up with the wrong group, the owner runs:

```bash
DATA_DIR=/corral/utexas/IRI26024/oopsiedata
find "$DATA_DIR" -user $USER ! -group G-829069 -exec chgrp -h G-829069 {} +
find "$DATA_DIR" -user $USER -type d ! -perm -g+s -exec chmod g+s {} +
```

## One-time setup

The member of `G-829069` who runs the nightly job does this once:

1. Make sure their SSH key can read `oopsie-backend` and push to `oopsie-docs` on GitHub.
2. Put the Hugging Face tokens next to the data, readable by the group only. `master.hf_token` is the read token the jobs use.

    ```bash
    cp tokens.yaml /corral/utexas/IRI26024/oopsiedata/tokens.yaml
    chgrp G-829069 /corral/utexas/IRI26024/oopsiedata/tokens.yaml
    chmod 640 /corral/utexas/IRI26024/oopsiedata/tokens.yaml
    ```

3. Install the cron entry: `scrontab slurm/daily.scron`.

`tokens.yaml` never goes into the repo. A job takes `master.hf_token` from it unless `HF_TOKEN` is set.

## The nightly job

`slurm/daily.scron` runs at 3:00 every day. It fetches the latest `main` of `oopsie-backend` into `$SCRATCH/oopsie/oopsie-backend` and runs `slurm/daily_oopsie.slurm` from there, which:

1. fetches the `oopsie-docs` site, which receives the website stats;
2. pins `numpy<2.4`, since 2.4 needs CPU features older nodes lack;
3. runs the pipeline target `all` with the LS6 paths, which syncs the Hub first;
4. commits and pushes `stats.json` to `oopsie-docs` if it changed;
5. resets the group and modes of the index, reports and logs;
6. fails if any repo failed to sync, even though the rest of the pipeline ran.

If `oopsie-docs` cannot be fetched, `stats.json` goes to the reports directory and is not published.

To start the job by hand, for example after a failure:

```bash
sbatch $SCRATCH/oopsie/oopsie-backend/slurm/daily_oopsie.slurm
sbatch -p normal $SCRATCH/oopsie/oopsie-backend/slurm/daily_oopsie.slurm   # if vm-small is busy
sinfo -S+P -o "%18P %10l %20F"                                              # free nodes per partition
```

## Checking a run

The log is `/corral/utexas/IRI26024/logs/oopsie-daily-<jobid>.log`. Each step is framed by `START`, `SUCCESS` or `FAILED` lines, and the last line says whether the job as a whole succeeded:

```
=== 2026-06-14T10:00:05-05:00 START: pipeline ===
  Syncing 18 repo(s) of OopsieData-Submissions into /corral/utexas/IRI26024/oopsiedata/OopsieData-Submissions
  ...
=== 2026-06-14T10:20:00-05:00 SUCCESS: pipeline completed in 1195s ===
=== 2026-06-14T10:20:00-05:00 SUCCESS: oopsie-daily completed in 1200s (job 12345) ===
```

The pipeline's `--explain` output in the log lists which stages ran and which came from the cache. `oopsiedata/.sync_report.json` lists the result of every repo's sync.

## Other targets

`slurm/pipeline.slurm` runs any target on a compute node with the same paths as the nightly job. Submit it from the staged checkout:

```bash
cd $SCRATCH/oopsie/oopsie-backend
sbatch slurm/pipeline.slurm training --set training_recipe=configs/training/droid_joint_velocity.yaml
sbatch slurm/pipeline.slurm imports
sbatch slurm/pipeline.slurm sync --hf-token <token> --set data_dir=/path/to/mirror
```

Everything after the target goes to `oopsie pipeline run`; `--hf-token` replaces the token from `tokens.yaml`.

## Running commands interactively

To run a command outside a job, stage a checkout and set the variables the jobs set:

```bash
B=$SCRATCH/oopsie/oopsie-backend
git clone --depth 1 --branch main git@github.com:oopsie-data/oopsie-backend.git $B   # or: git -C $B pull
export UV_PYTHON_INSTALL_DIR=$SCRATCH/oopsie/tools/uv-python UV_CACHE_DIR=$SCRATCH/oopsie/uv-cache HF_HOME=$SCRATCH/oopsie/hf_home
export HF_TOKEN=<master.hf_token from /corral/utexas/IRI26024/oopsiedata/tokens.yaml>
cd $B
```

On `vm-small` nodes, numpy must stay below 2.4: `uv add "numpy<2.4" && uv sync --reinstall-package numpy`.

Every path the jobs use can be overridden with an environment variable: `OOPSIE_ROOT`, `OOPSIE_DATA_DIR`, `OOPSIE_INDEX_DIR`, `OOPSIE_REPORTS_DIR`, `OOPSIE_LOG_DIR`, `OOPSIE_TOKENS`, `OOPSIE_STAGE`, `OOPSIE_NODE_STAGE`, `OOPSIE_BACKEND`, `OOPSIE_DOCS`, `OOPSIE_DOCS_REMOTE`, `OOPSIE_DOCS_BRANCH`, `OOPSIE_GIT_EMAIL` and `UV`.

## When something fails

| Symptom | Cause and fix |
|---|---|
| the job ends with "repos failed to sync" | a repo's download failed; `.sync_report.json` names it and the error. The rest of the pipeline ran on the data that synced. Rerun the job, or sync that repo by hand with `--only`. |
| a stage fails because its settings changed | a written-once output (chosen proposal, train sample, assignments) no longer matches `configs/eval_split.yaml`. Revert the setting, or, if the output really should change and no annotator has started on it, delete the output and its `.inputs` file. |
| `Disk quota exceeded` | files of the wrong group; see [permissions](#permissions). If Corral is out of files, keep the sync record on `$SCRATCH` with `--state`. |
| QA was interrupted or hit the time limit | its results are saved every five minutes and on SIGTERM; the next run continues with the remaining episodes. |
| a code change did not take effect | Hamilton caches a stage by its own source only; rerun the target with `--no-cache`. |
| numpy fails with an illegal instruction | the node's CPU is too old for numpy 2.4; pin `numpy<2.4` as above. |
| `stats.json` was not published | `oopsie-docs` could not be fetched or pushed; the file is in the reports directory. |

In a multi-node job, the index build, QA and training split their episodes across nodes and write intermediate results to shard directories (`<index dir>/shards/index`, `<index dir>/shards/qa`, `<training work dir>/<recipe>/shards`). The LS6 jobs use a single node, so these directories normally stay empty.
