# Hub sync

Labs upload to Hugging Face, but the pipeline reads local files. The sync keeps a local mirror of the organization's repos up to date, and is the first stage of every nightly run.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `hub_mirror` | `sync` | nothing; runs on every invocation | h5 and video files under `<data_dir>/OopsieData-Submissions/`, `.sync_state.parquet`, `.sync_report.json` |
| `hub_upload_times` | `index` | `hub_mirror`, `episode_index` | `hub_uploaded_at`, `hub_upload_commit`; git histories in `<data_dir>/.hub_history/` |

With `use_hub: false`, both stages do nothing, so the pipeline can run offline on an existing mirror.

## How the sync works

The sync lists every repo of the organization at its current commit and compares each file with its record in `<data_dir>/.sync_state.parquet`, which holds every synced file's Hub blob id, sha256, local size and modification time:

| Local file | Action |
|---|---|
| recorded, unchanged on the Hub, same size and modification time | kept; the file is not read |
| changed on the Hub | downloaded |
| on disk but not recorded, or modified since | adopted if its hash matches the Hub, else downloaded |
| missing | downloaded |
| recorded but gone from the Hub | reported; deleted only with `--delete-removed` |

The mirror is therefore a superset of the Hub: a lab that deletes an episode does not delete it locally. Archive and infra repos (`configs/repositories.yaml`), such as the annotation repos, are skipped unless `--only` names them.

Downloads go to a temporary name and are renamed into place, so an interrupted sync never leaves a partial file under the real name. The sync record is written after every repo, so a rerun resumes where the last one stopped. A repo that fails is recorded in the report and does not stop the others or the pipeline.

The sync writes nothing else next to the data. `huggingface_hub` keeps a metadata and a lock file per downloaded file in `.cache/huggingface` folders; the sync deletes them once a repo is fully recorded. This matters on Corral, which limits the number of files per allocated terabyte.

## The sync report

`<data_dir>/.sync_report.json` summarizes the last sync: totals and, per repo, the counts of downloaded, adopted and unchanged files, the files removed on the Hub, failures, and the Hub commit of every repo that synced completely. `repos_not_on_hub` lists local repo folders the Hub no longer has. The nightly job fails at the end when any repo failed, after the rest of the pipeline has run.

A clean report pins the data at those commits, which is what a [data release](release.md) is made from.

```
Syncing 18 repo(s) of OopsieData-Submissions into /corral/utexas/IRI26024/oopsiedata/OopsieData-Submissions
[1/18] OopsieData-Submissions/SomeLab: downloaded 12, adopted 0, unchanged 450, removed on Hub 0, failed 0 (35s)
...
Downloaded 12 file(s), adopted 0, unchanged 37041, removed on Hub 3, repos failed 0
```

## Running it by hand

```bash
export HF_TOKEN=<read token>
uv run oopsie ingest sync --data-dir /corral/utexas/IRI26024/oopsiedata --group G-829069
```

On LS6, run it under `sg G-829069` so that new files belong to the project group (see [permissions](../operations/ls6.md#permissions)).

| Option | Effect |
|---|---|
| `--only REPO ...` | sync only these repos |
| `--smallest N` | sync only the N smallest contributor repos, for a small local test mirror |
| `--dry-run` | report what would be downloaded and write nothing |
| `--no-download` | check and adopt existing files and delete `.cache` folders, without downloading |
| `--state PATH` | keep the sync record and report elsewhere, e.g. on `$SCRATCH` when Corral is out of files |
| `--scan-local` | also report local files and repo folders the Hub does not have |
| `--delete-removed` | delete local files the Hub no longer has |
| `--release NAME` | download a [data release](release.md) and sync each repo at its pinned commit |
| `--revisions FILE` | sync each repo at the commit a snapshot file gives |
| `--split test`, `--split train` | with `--release`: download only the release index and that split's h5 and video files ([download a release](../using/download.md#only-the-test-or-the-train-set)); the report records the split, and `release publish` refuses it |
| `--keep-hf-cache` | keep the `.cache/huggingface` folders |
| `--group GROUP` | set this group on every file and folder the sync creates |
| `--parallel-repos N` | repos synced at the same time (default 4) |
| `--workers N` | parallel downloads within a repo (default 4) |

`--release` and `--revisions` refuse to write into the shared mirror, since they would move it back to older commits.

## Upload times

`hub_uploaded_at` is the date of the commit that first added an episode's h5 to the default branch of its repo, and `hub_upload_commit` is that commit. A file added through a merged pull request counts from the merge commit. The history of each repo, commits and trees without file contents, is kept as a bare git clone in `<data_dir>/.hub_history/<repo>.git` and fetched incrementally.

A repo whose fetch fails keeps its earlier values, and a repo whose history was squashed dates every file to the squash commit. No stage depends on these columns.

```bash
uv run oopsie ingest upload-times --data-dir <mirror> --index episode_index.parquet
```
