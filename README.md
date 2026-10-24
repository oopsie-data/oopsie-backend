# oopsie-backend

All oopsiedata processing code lives here. Every stage reads and extends one per-episode parquet index; [Hamilton](https://hamilton.apache.org/) runs the stages and reruns only what changed.

```bash
uv sync
uv run oopsie                                   # list stages and commands
uv run oopsie pipeline run all --explain        # the full pipeline
```

## Layout

| Package | Stage |
|---|---|
| `oopsie_backend/ingest` | Sync the Hugging Face org to the local mirror |
| `oopsie_backend/index` | Build the episode index and read or update its columns |
| `oopsie_backend/preqa` | Checks before QA: duplicate episodes and duplicate videos |
| `oopsie_backend/qa` | QA checks, one file per check, reports, spot checks and lab feedback |
| `oopsie_backend/postqa` | Critical episodes |
| `oopsie_backend/stats` | Public and internal stats |
| `oopsie_backend/splits` | Eval split proposals, the chosen split, and the annotated train sample |
| `oopsie_backend/annotation` | Annotator assignment, bundles, Hugging Face distribution and collection |
| `oopsie_backend/export` | Export to RLDS |
| `oopsie_backend/conversion` | Converters from external datasets |
| `oopsie_backend/pipeline` | Stage registry, targets and the Hamilton driver |
| `oopsie_backend/common` | Repo kinds, h5 decoding, video paths, shared I/O |

`oopsie_backend/scripts/` holds aliases for the earlier command paths.

## Documentation

| Topic | Doc |
|-------|-----|
| Pipeline | [docs/pipeline.md](docs/pipeline.md) |
| Episode index | [docs/episode-index.md](docs/episode-index.md) |
| Sync and stats | [docs/sync-and-stats.md](docs/sync-and-stats.md) |
| QA spot checks | [docs/qa-spot-checks.md](docs/qa-spot-checks.md) |
| Action checks | [docs/action-checks.md](docs/action-checks.md) |
| Video checks | [docs/video-checks.md](docs/video-checks.md) |
| Eval split proposals | [docs/eval-split.md](docs/eval-split.md) |
| Annotation coordinator | [docs/annotation-coordinator.md](docs/annotation-coordinator.md) |
| Annotator guide | [docs/annotator-guide.md](docs/annotator-guide.md) |
