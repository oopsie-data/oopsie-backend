# Episode index

The index build creates the [episode index](../architecture.md#the-episode-index): it finds every episode in the mirror, reads its h5 metadata and the Hub's hashes, and writes one row per episode. All later stages add their columns to these rows.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `episode_index` | `index` | `mirror_state`, which changes when an h5 file or `configs/repositories.yaml` or `configs/imports.yaml` changes | the core columns; creates the index if it is missing |

## What it reads

Episodes are discovered under `<data_dir>/OopsieData-Submissions/<repo>/` and, for local imports, under `<data_dir>/imports/<name>/`, whose dataset id is `imports/<name>`. Archive and infra repos are skipped ([repo kinds](../architecture.md#repos)).

For each h5 file, the build reads the root attributes (lab, operator, instruction, timestamp), the robot profile (robot, gripper, policy, control frequency, action space), the lab's own annotations, the camera list, and the action and state arrays, which it hashes into `content_hash`. It also lists every repo on the Hub to record the Hub's sha256 and git object id of each h5 and video. Lab ids are mapped to canonical names with `configs/lab_ids.yaml`, which fixes labs that recorded misspelled ids.

A file that cannot be read gets a row with `read_error` set and null h5 columns, so it stays visible in the index and becomes [critical](critical-episodes.md). A file that disappears from the mirror keeps its row with `on_disk = false`.

The [column reference](../reference/index-columns.md#core) describes every core column.

## Incremental builds

The build rereads a file only when its size or modification time differs from its row, or when it has no row yet. A newly added core column is the exception: the index then lacks the column, and every file is read once.

When the Hub listing of a repo fails, the repo keeps its earlier hashes and `on_hub` values. `--no-hub` skips the listing.

```bash
uv run oopsie index build --data-dir <mirror> --index episode_index.parquet [--workers N] [--no-hub]
```

In a multi-node Slurm job, the build splits the files across the nodes. Each node writes its rows to `<index dir>/shards/index`, and only the coordinating process writes the index.
