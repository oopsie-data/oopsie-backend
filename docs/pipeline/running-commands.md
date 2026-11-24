# Running commands

`oopsie <group> <command>` runs one job on its own: the code a stage calls, with options of your choosing. Nothing is cached and no other stage runs first, so it suits debugging, one-off runs on another index or mirror, and the steps that are not stages at all. `--help` after a command prints its options, and the [command reference](../reference/commands.md) lists every command.

```bash
uv run oopsie                                                    # every command group and command
uv run oopsie index build --help                                 # one command's options
uv run oopsie stats compute-local --index episode_index.parquet  # run it
```

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

> **TODO:** The code of the `preqa` and `postqa` stages (`oopsie_backend/preqa/`, `oopsie_backend/postqa/`) is not registered in `cli.py` yet, so it runs as `uv run python -m oopsie_backend.preqa.duplicate_episodes`, `….preqa.duplicate_videos` and `….postqa.critical`.

