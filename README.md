# oopsie-backend

All Oopsie-Data processing code lives here. Every stage reads and extends one per-episode parquet index; [Hamilton](https://hamilton.apache.org/) runs the stages and reruns only what changed.

There are two things you can do:

1. **Download the data and use it** (this is the quickstart): get a release (the full dataset or a part of it) and its episode index. See [Download a release](docs/using/download.md).
2. **Run the entire pipeline** (more heavy): sync the labs' uploads, build the index, run QA, the eval split, the annotation and the releases. See [Pipeline](docs/pipeline/index.md).


| Topic | Page |
|---|---|
| Download a release | [docs/using/download.md](docs/using/download.md) |
| Reading the index | [docs/using/reading-the-index.md](docs/using/reading-the-index.md) |
| Pipeline overview: the mirror, the index, the rules | [docs/pipeline/index.md](docs/pipeline/index.md) |
| Running the pipeline: stages, targets, inputs | [docs/pipeline/running.md](docs/pipeline/running.md) |
| LS6 runbook | [docs/operations/ls6.md](docs/operations/ls6.md) |
| Index columns, configs, commands | [docs/reference/](docs/reference/index-columns.md) |
| Code layout, adding a stage, browsing the docs locally | [docs/development.md](docs/development.md) |


You can also browse the documentation as an MkDocs site, built from `docs/` and starting at [docs/index.md](docs/index.md):

```bash
uv run --group docs mkdocs serve     # then open http://127.0.0.1:8000
```
