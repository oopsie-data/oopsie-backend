# Oopsie backend

Oopsie-Data collects robot episodes, successes and failures, from many labs. Each lab uploads its episodes as HDF5 files with camera videos to its own dataset repo in the `OopsieData-Submissions` organization on Hugging Face. The backend turns these uploads into curated data: it mirrors the repos, catalogs every episode, checks the data, holds out an evaluation set, distributes episodes to annotators, collects their labels and materializes training sets.

All of this runs as one pipeline on TACC's LS6 cluster. A nightly job brings the pipeline up to date with whatever the labs uploaded that day, and every result lands in a single table, the [episode index](architecture.md#the-episode-index), with one row per episode.

## Where to start

| You want to | Read |
|---|---|
| understand how the pieces fit together | [Architecture](architecture.md) |
| run the pipeline or understand a stage | [Pipeline](pipeline/index.md) |
| look up an index column or a config file | [Index columns](reference/index-columns.md), [Configs](reference/configs.md) |
| run or repair the nightly job on LS6 | [LS6 runbook](operations/ls6.md) |
| coordinate the annotators | [Annotation](pipeline/annotation/index.md) |
| annotate episodes | [Annotator guide](pipeline/annotation/annotator-guide.md) |
| build a training set | [Training data](pipeline/training-data.md) |
| add a stage, a QA check or a converter | [Development](development.md) |

## Setup

The backend is a Python package managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                          # install
uv run oopsie                    # list every command
uv run oopsie pipeline run --list
```

The dataset analysis needs the optional `analysis` extra (`uv sync --extra analysis`). To work on these docs, serve them locally:

```bash
uv run --group docs mkdocs serve
```
