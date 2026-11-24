# oopsie-backend

All oopsiedata processing code lives here. Every stage reads and extends one per-episode parquet index; [Hamilton](https://hamilton.apache.org/) runs the stages and reruns only what changed.

To download the data, ask the Oopsie-Data team for a read-only Hugging Face token, then:

```bash
uv sync
export HF_TOKEN=<your read-only token>
uv run oopsie ingest release list                                       # the published releases
uv run oopsie ingest sync --release 2026-11-08 --data-dir /path/to/data
```

To run the pipeline (the Oopsie-Data team):

```bash
uv sync
uv run oopsie                                   # list command groups and commands
uv run oopsie pipeline run all --explain        # the full pipeline
```

## Layout

| Package | Contents |
|---|---|
| `oopsie_backend/ingest` | Sync the Hugging Face org to the local mirror |
| `oopsie_backend/index` | Build the episode index and read or update its columns |
| `oopsie_backend/preqa` | Checks before QA: duplicate episodes and duplicate videos |
| `oopsie_backend/qa` | QA checks, one file per check, reports, spot checks and lab feedback |
| `oopsie_backend/postqa` | Critical episodes |
| `oopsie_backend/stats` | Public and internal stats |
| `oopsie_backend/splits` | Eval split proposals, the chosen split, and the annotated train sample |
| `oopsie_backend/annotation` | Annotator assignment, bundles, Hugging Face distribution and collection |
| `oopsie_backend/training` | Training data recipes: selection, unification, LeRobot, RLDS and HDF5 output |
| `oopsie_backend/export` | Export to RLDS |
| `oopsie_backend/conversion` | Converters from external datasets and the imports stage |
| `oopsie_backend/pipeline` | Stage registry, targets and the Hamilton driver |
| `oopsie_backend/common` | Repo kinds, h5 decoding, video paths, episode staging, shared I/O |

`oopsie_backend/scripts/` holds aliases for the earlier command paths.

## Documentation

The documentation is a MkDocs site in `docs/`, starting at [docs/index.md](docs/index.md). To browse it locally:

```bash
uv run --group docs mkdocs serve
```

| Topic | Page |
|---|---|
| Download a release | [docs/using/download.md](docs/using/download.md) |
| Reading the index | [docs/using/reading-the-index.md](docs/using/reading-the-index.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Pipeline and stages | [docs/pipeline/index.md](docs/pipeline/index.md) |
| LS6 runbook | [docs/operations/ls6.md](docs/operations/ls6.md) |
| Index columns, configs, commands | [docs/reference/](docs/reference/index-columns.md) |
| Development | [docs/development.md](docs/development.md) |
