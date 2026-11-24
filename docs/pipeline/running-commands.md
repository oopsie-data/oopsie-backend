# Running commands

`oopsie <group> <command>` runs one job on its own: the code a stage calls, with options of your choosing. Nothing is cached and no other stage runs first, so it suits debugging, one-off runs on another index or mirror, and the steps that are not stages at all. `--help` after a command prints its options. [All commands](#all-commands) lists every command by group.

```bash
uv run oopsie                                                    # every command group and command
uv run oopsie index build --help                                 # one command's options
uv run oopsie stats compute-local --index episode_index.parquet  # run it
```

## All commands

> **TODO:** The `preqa` and `postqa` code (duplicate episodes, duplicate videos, critical episodes) is not registered in `cli.py` yet. Run it as `uv run python -m oopsie_backend.preqa.duplicate_episodes`, `….preqa.duplicate_videos` and `….postqa.critical`.

### Pipeline

| Command | Does |
|---|---|
| `pipeline run [TARGET]` | runs a target and every stage it needs ([running targets](running-targets.md)) |

### Ingest

| Command | Does |
|---|---|
| `ingest sync` | syncs the mirror with the Hub, or downloads a release ([Hub sync](sync.md)) |
| `ingest upload-times` | writes the Hub upload times to the index |
| `ingest release publish`, `ingest release list` | publishes or lists [data releases](release.md) |

### Index

| Command | Does |
|---|---|
| `index build` | builds or refreshes the index's core columns ([episode index](episode-index.md)) |
| `index camera-names` | writes the unified camera names ([camera names](camera-names.md)) |
| `index camera-sheet` | writes a page with a frame of every dataset's cameras, for checking the mapping |
| `index embodiments` | writes the unified arms and end effectors ([embodiments](embodiments.md)) |
| `index policies` | writes the policy families and classes ([policies](policies.md)) |

### QA

| Command | Does |
|---|---|
| `qa run` | runs the QA checks without a current result ([QA](qa/index.md)) |
| `qa render-report` | writes `action_qa.html` from the index |
| `qa serve-report` | serves the report with videos and evidence, behind a password |
| `qa render-lab-feedback` | writes the per-lab feedback text from the QA results |
| `qa select-spot-checks` | selects episodes per lab and writes review bundles ([spot checks](qa/spot-checks.md)) |
| `qa merge-notes` | merges returned spot-check notes into a table or the index |

### Stats

| Command | Does |
|---|---|
| `stats compute-local` | computes the public and internal stats from the index ([stats](stats.md)) |
| `stats compute-hf` | computes the stats by streaming repos from the Hub |
| `stats render-public`, `stats render-internal` | renders the stats pages from their JSON |
| `stats analysis {index,annotations,features,data}` | computes the [dataset analysis](analysis.md) |

### Splits

| Command | Does |
|---|---|
| `splits propose` | writes eval split proposals and their report ([eval split](eval-split.md)) |
| `splits package` | zips a proposal's eval episodes for review |
| `splits record` | writes a proposal's splits to the index |
| `splits sample-train` | samples train episodes for annotation ([train sample](train-sample.md)) |

### Annotation

| Command | Does |
|---|---|
| `annotation pipeline <command>` | runs the Hugging Face distribution and collection: `stage`, `publish`, `pull`, `verify`, `collect`, `status`, `index`, `release` ([annotation](annotation/index.md)) |
| `annotation assign` | splits a proposal into per-annotator lists |
| `annotation stage` | lays out batches as the folder tree of the batch repo |
| `annotation package` | zips each annotator's list with its files, for distribution without Hugging Face |
| `annotation collect` | collects annotations from returned zip bundles into a store |
| `annotation flags` | writes the [annotation flags](annotation/flags.md) |

### Training and export

| Command | Does |
|---|---|
| `training materialize` | builds a training set from a recipe ([training data](training-data.md)) |
| `export to-rlds` | converts Oopsie HDF5 episodes to a DROID-style RLDS dataset |

### Conversion

| Command | Does |
|---|---|
| `convert imports` | converts the local imports of `configs/imports.yaml` ([imported datasets](imports.md)) |
| `convert soar`, `convert roboarena`, `convert ar-aloha`, `convert rlds` | runs a single converter |

### Tools

| Command | Does |
|---|---|
| `tools inspect-hdf5` | prints the tree of an episode file |
| `tools visualize` | plots an episode's trajectories and frames, or a directory's summary |

## Command groups, stages and targets

The same code can be reached in two ways, organized differently:

| | Command group | Target |
|---|---|---|
| Run as | `oopsie <group> <command>` | `oopsie pipeline run <target>` |
| Contains | commands | stages |
| Organized by | code folder: `oopsie_backend/<group>/` | the order of a run |
| Defined in | `COMMANDS` in `oopsie_backend/cli.py` | `TARGETS` in `oopsie_backend/pipeline/stages.py` |

A stage is a function in a folder's `stages.py` that calls one of that folder's commands with the pipeline's inputs. Running the command yourself does the same work with options of your choosing, but without the cache or the stages it depends on.

Many command groups and targets share a name because a step of the run mostly uses one folder's code, but their contents differ. The `index` name, for example:

| Command (`oopsie index …`) | Stage that calls it | Target of the stage |
|---|---|---|
| `index build` | `episode_index` | `index` |
| `index camera-names` | `unified_camera_names` | `preqa` |
| `index embodiments` | `unified_embodiments` | `preqa` |
| `index policies` | `unified_policies` | `preqa` |
| `index camera-sheet` | none; a helper for checking the camera mapping | |
| `ingest upload-times` | `hub_upload_times` | `index` |

The camera, embodiment and policy stages write index columns, so their code lives in `index/`, but they wait for the duplicates, so they run in `preqa`. Likewise the `ingest` folder's three stages belong to three targets: `hub_mirror` to `sync`, `hub_upload_times` to `index` and `data_release` to `release`. There is no `ingest` target.

Some targets have no command group of the same name (`all`, `preqa`, `postqa`, `proposals`), and some command groups have no target (`convert`, `export`, `tools`).
