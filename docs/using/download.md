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

```bash
uv run oopsie ingest sync --release 2026-11-08 --data-dir /path/to/data
```

This downloads the release's index, then every episode file at its pinned commit. Every file is checked against the hash the Hub records for it.

<!-- TODO: document `--split test` and `--split train` once implemented. -->

## What you get

```text
/path/to/data/
├── releases/2026-11-08/
│   ├── episode_index.parquet     # one row per episode
│   ├── snapshot.json             # the pinned commit of every repo
│   ├── manifest.json             # backend commit and the sha256 of every release file
│   ├── annotation/               # eval split and annotation assignments
│   └── configs/                  # the backend configs the release was built with
├── annotation/                   # copied from the release where missing
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
