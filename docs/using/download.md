# Download a release

A release is a frozen, tagged version of the dataset, named by its date, e.g. `2026-11-08`. It pins the Hugging Face commit of every lab's repo and ships the [episode index](reading-the-index.md) built from exactly those files, so everyone who downloads the same release gets the same episodes and the same index.

The episode files stay in the labs' repos in the [`OopsieData-Submissions`](https://huggingface.co/OopsieData-Submissions) organization; the release records which commit of each repo to download.

## Access

The releases and the labs' repos are private. Ask the Oopsie-Data team for a read-only Hugging Face token, and set it in your shell:

```bash
export HF_TOKEN=<your read-only token>
```

## Install

The download is a command of the backend, a Python package managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/oopsie-data/oopsie-backend.git
cd oopsie-backend
uv sync
```

## List the releases

```bash
uv run oopsie ingest release list
```

This prints each release's name and the commit of the release repo it points to.

## Download

### The whole release

```bash
uv run oopsie ingest sync --release 2026-11-08 --data-dir /path/to/data
```

This downloads the release's index, then every file of every lab repo at its pinned commit.

### Only the test or the train set

```bash
uv run oopsie ingest sync --release 2026-11-08 --split test --data-dir /path/to/data
```

`--split` downloads the release's index, snapshot and manifest, and then only the h5 and video files of the episodes in that split:

| `--split` | Episodes | `eval_split` in the index | Size (`2026-11-08`) |
|---|---|---|---|
| `test` | 2,000 | `id` or `od_task_id_scene` | about 30 GiB |
| `train` | 144,210 | null | about 1.6 TiB |
| none: the whole release | 147,076 | any | about 1.7 TiB |

`excluded` episodes belong to neither split: they come from the same held-out labs and tasks as the test set, so never train on them ([test and train](reading-the-index.md#test-and-train)). Each file is checked against the blob the index recorded for it, so the files always match the index.

To get more later, run the command again with another split or without `--split`, into the same folder: files you already have are kept and only the rest downloads.

### Only the files listed in an index

Use `oopsie ingest download` with an episode-index Parquet or a filtered copy.

```bash
uv run oopsie ingest download \
  --index data/releases/2026-11-08/episode_index_sample.parquet \
  --data-dir data/releases/2026-11-08/sample_data \
  --revisions data/releases/2026-11-08/snapshot.json
```

The Parquet needs `dataset_repo` (`namespace/repo`) and `repo_path` (a file path
inside the repo). If it has a `videos` list with a `path` field per video, the
command downloads those files too. Files go under `<data-dir>/<namespace>/<repo>/`
with their original paths. The command skips local imports, verifies file
hashes, and reuses matching files on reruns. It uses your Hugging Face login
or `HF_TOKEN` for private datasets.

Pass the release's `snapshot.json` to `--revisions` to use its pinned commits.
Without a snapshot, the command requests files from the repos' default branches.

## What you get

```text
/path/to/data/
├── releases/2026-11-08/
│   ├── episode_index.parquet     # one row per episode
│   ├── snapshot.json             # the pinned commit of every repo
│   ├── manifest.json             # backend commit and the sha256 of every release file
│   ├── annotation/               # eval split and annotation assignments (not with --split)
│   └── configs/                  # the backend configs the release was built with (not with --split)
├── annotation/                   # copied from the release where missing (not with --split)
└── OopsieData-Submissions/
    └── <lab repo>/
        ├── <episode>.h5          # actions, states and metadata
        └── <episode>_<camera>.mp4
```

The [Reading the index](reading-the-index.md) page explains how to go from an index row to its files.

## Resuming

Run the same command again to resume an interrupted download: files already downloaded and unchanged are skipped. The sync keeps a record of what it downloaded in `.sync_state.parquet` and a summary of the last run in `.sync_report.json`, both in the data folder.

## Limits

- A release depends on the labs' repos: it breaks if a lab deletes its repo or squashes its history.
- The download refuses to write into the team's shared mirror on LS6; use your own folder.
