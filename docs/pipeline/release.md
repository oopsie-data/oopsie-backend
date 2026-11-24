# Data releases

The mirror and the index change every night, so results computed from them cannot be reproduced later. A data release pins one state of the data: the Hub commit of every repo, the index built from exactly those files, and the annotation decisions made on them. Anyone with access can download the release and get the same episodes and the same index.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `data_release` (manual) | `release` | nothing; publishes the mirror and index as they are | a folder and a tag `<name>` in `OopsieData-Submissions/oopsiedata-releases` |

## Contents

Each release is a folder of the private repo [`OopsieData-Submissions/oopsiedata-releases`](https://huggingface.co/datasets/OopsieData-Submissions/oopsiedata-releases), tagged with the same name, by default the sync date:

| File | Contents |
|---|---|
| `snapshot.json` | a clean [sync report](sync.md#the-sync-report), with the Hub commit of every synced repo |
| `episode_index.parquet` | the index built from exactly that data |
| `annotation/` | the chosen proposal, train sample, assignments, annotator list, proposals and annotation stores, with their `.inputs` records |
| `configs/` | the backend's configs |
| `manifest.json` | the backend commit, the commits of the annotation repos and the sha256 of every file |

The episode files themselves stay in the labs' repos; the snapshot pins the commits to download them at.

## Publishing

The sync report records a repo's commit only when the repo synced completely, so a release starts with a clean sync. Sync, update the index, then copy the report and publish it:

```bash
cp $DATA_DIR/.sync_report.json $DATA_DIR/snapshots/$(date -I).json
uv run oopsie ingest release publish --snapshot $DATA_DIR/snapshots/$(date -I).json
```

or, through the pipeline, `uv run oopsie pipeline run release --set release_snapshot=<snapshot>`.

Publishing refuses unless:

- the sync was clean;
- every index episode is in the snapshot, with the sha256 the index recorded;
- the sync record lists exactly the files of each pinned commit (`--skip-hub-check` skips this check).

`--dry-run` runs the checks and builds the release without uploading. Releases are never replaced, and `oopsie ingest release list` lists them.

## Downloading

Users download a release with `oopsie ingest sync --release <name>`, described in [Download a release](../using/download.md). `--revisions <snapshot.json>` instead syncs each repo at the commits of a local snapshot file, without fetching a release.
