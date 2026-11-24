# oopsie-backend

All Oopsie-Data processing code lives here. Every stage reads and extends one per-episode parquet index; [Hamilton](https://hamilton.apache.org/) runs the stages and reruns only what changed.

There are two things you can do:

1. **Get the data** *(start here)*: download a release, the full dataset or a part of it, and its episode index. It takes minutes, and it is what most people need.
2. **Run the full pipeline** *(Oopsie-Data team only)*: rebuild everything from the labs' uploads: sync, index, QA, eval split, annotation and releases. It is heavy and needs LS6 and the team's token.

| Topic | Page |
|---|---|
| **Get the data** *(start here)* | [docs/using/download.md](docs/using/download.md) |
| **Run the full pipeline** *(Oopsie-Data team only)* | [docs/pipeline/quickstart.md](docs/pipeline/quickstart.md) |
| LS6 runbook: the nightly job, permissions, fixing failed runs | [docs/operations/ls6.md](docs/operations/ls6.md) |
| Reference: index columns and configs | [docs/reference/](docs/reference/index-columns.md) |
| Development: code layout, adding a stage, browsing the docs locally | [docs/development.md](docs/development.md) |

You can also browse the documentation as an MkDocs site, built from `docs/` and starting at [docs/index.md](docs/index.md):

```bash
uv run --group docs mkdocs serve     # then open http://127.0.0.1:8000
```
