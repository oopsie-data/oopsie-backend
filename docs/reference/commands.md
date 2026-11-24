# Commands

Every command runs as `uv run oopsie <group> <command>`, and `--help` after a command prints its options. `uv run oopsie` alone lists them. A command group is named after the code folder its commands live in; it is not a pipeline stage or target, see [Commands, stages and targets](../pipeline/index.md#commands-stages-and-targets).

## For users

| Command | Does |
|---|---|
| `ingest release list` | lists the published releases |
| `ingest sync --release NAME --data-dir DIR` | downloads a release ([Download a release](../using/download.md)) |
| `tools inspect-hdf5 FILE` | prints the tree of an episode file |

Everything below is for running the pipeline.

> **TODO:** The `preqa` and `postqa` code (duplicate episodes, duplicate videos, critical episodes) is not registered in `cli.py` yet. Run it as `uv run python -m oopsie_backend.preqa.duplicate_episodes`, `….preqa.duplicate_videos` and `….postqa.critical`.

## Pipeline

| Command | Does |
|---|---|
| `pipeline run [TARGET]` | runs a target and every stage it needs ([pipeline](../pipeline/index.md#running)) |

## Ingest

| Command | Does |
|---|---|
| `ingest sync` | syncs the mirror with the Hub, or downloads a release ([Hub sync](../pipeline/sync.md)) |
| `ingest upload-times` | writes the Hub upload times to the index |
| `ingest release publish`, `ingest release list` | publishes or lists [data releases](../pipeline/release.md) |

## Index

| Command | Does |
|---|---|
| `index build` | builds or refreshes the index's core columns ([episode index](../pipeline/episode-index.md)) |
| `index camera-names` | writes the unified camera names ([camera names](../pipeline/camera-names.md)) |
| `index camera-sheet` | writes a page with a frame of every dataset's cameras, for checking the mapping |
| `index embodiments` | writes the unified arms and end effectors ([embodiments](../pipeline/embodiments.md)) |
| `index policies` | writes the policy families and classes ([policies](../pipeline/policies.md)) |

## QA

| Command | Does |
|---|---|
| `qa run` | runs the QA checks without a current result ([QA](../pipeline/qa/index.md)) |
| `qa render-report` | writes `action_qa.html` from the index |
| `qa serve-report` | serves the report with videos and evidence, behind a password |
| `qa render-lab-feedback` | writes the per-lab feedback text from the QA results |
| `qa select-spot-checks` | selects episodes per lab and writes review bundles ([spot checks](../pipeline/qa/spot-checks.md)) |
| `qa merge-notes` | merges returned spot-check notes into a table or the index |

## Stats

| Command | Does |
|---|---|
| `stats compute-local` | computes the public and internal stats from the index ([stats](../pipeline/stats.md)) |
| `stats compute-hf` | computes the stats by streaming repos from the Hub |
| `stats render-public`, `stats render-internal` | renders the stats pages from their JSON |
| `stats analysis {index,annotations,features,data}` | computes the [dataset analysis](../pipeline/analysis.md) |

## Splits

| Command | Does |
|---|---|
| `splits propose` | writes eval split proposals and their report ([eval split](../pipeline/eval-split.md)) |
| `splits package` | zips a proposal's eval episodes for review |
| `splits record` | writes a proposal's splits to the index |
| `splits sample-train` | samples train episodes for annotation ([train sample](../pipeline/train-sample.md)) |

## Annotation

| Command | Does |
|---|---|
| `annotation pipeline <command>` | runs the Hugging Face distribution and collection: `stage`, `publish`, `pull`, `verify`, `collect`, `status`, `index`, `release` ([annotation](../pipeline/annotation/index.md)) |
| `annotation assign` | splits a proposal into per-annotator lists |
| `annotation stage` | lays out batches as the folder tree of the batch repo |
| `annotation package` | zips each annotator's list with its files, for distribution without Hugging Face |
| `annotation collect` | collects annotations from returned zip bundles into a store |
| `annotation flags` | writes the [annotation flags](../pipeline/annotation/flags.md) |

## Training and export

| Command | Does |
|---|---|
| `training materialize` | builds a training set from a recipe ([training data](../pipeline/training-data.md)) |
| `export to-rlds` | converts Oopsie HDF5 episodes to a DROID-style RLDS dataset |

## Conversion

| Command | Does |
|---|---|
| `convert imports` | converts the local imports of `configs/imports.yaml` ([imported datasets](../pipeline/imports.md)) |
| `convert soar`, `convert roboarena`, `convert ar-aloha`, `convert rlds` | runs a single converter |

## Tools

| Command | Does |
|---|---|
| `tools inspect-hdf5` | prints the tree of an episode file |
| `tools visualize` | plots an episode's trajectories and frames, or a directory's summary |
