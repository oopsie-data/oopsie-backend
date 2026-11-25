# Development

The backend is a uv project; dataframes use polars. New functionality belongs in the pipeline as a stage, not in a one-off script, so that it runs nightly, is cached and leaves its results in the index.

## Layout

| Package | Contents |
|---|---|
| `ingest` | Hub sync, upload times, data releases |
| `index` | the index build, the index store, camera names, embodiments, policies |
| `preqa` | duplicate episodes and videos |
| `qa` | QA checks, one file per check, reports, spot checks, lab feedback |
| `postqa` | critical episodes |
| `stats` | public and internal stats, dataset analysis |
| `splits` | eval split proposals, the chosen split, the train sample |
| `annotation` | assignments, bundles, Hugging Face distribution, collection, flags |
| `training` | recipes: selection, unification, output formats |
| `export` | RLDS export |
| `conversion` | converters for external datasets, the imports stage |
| `pipeline` | the stage registry, targets and the Hamilton driver |
| `common` | repo kinds, h5 decoding, video paths, staging, shards, shared I/O |

`oopsie_backend/scripts/` holds aliases for the earlier command paths.

`oopsie_backend/cli.py` maps `oopsie <group> <command>` to modules; a command group is usually named after its package (`convert` runs `conversion`). Each command is a module with a `main(argv)` and its own argument parser.

## Adding a stage

A stage is a function in `<package>/stages.py`, registered with `@stage`:

```python
@stage
def duplicate_videos(duplicate_episodes: str, index_path: str) -> str:
    check_rc(dv.main(["--index", index_path]), "duplicate videos")
    return columns_fingerprint(Path(index_path), dv.COLUMNS)
```

- **Parameters** name the stages it depends on and the inputs it needs from `configs/pipeline.yaml`. Hamilton derives the order from them.
- **Writes** go through `index.store.update_columns`, and only to the stage's own columns. Use `clear_missing=True` when the stage recomputes its whole output, so that episodes it no longer covers are reset.
- **The return value** is a fingerprint of what the stage wrote: `columns_fingerprint` for index columns, `files_fingerprint` for files. Stages after it rerun only when it changes.
- **`@stage(always_run=True)`** marks a watcher that hashes outside state, such as a config file. Give every new config a watcher, so that a change to it reruns the stage.
- **`@stage(manual=True)`** keeps the stage out of `all`.
- **`write_once`** creates an output once and fails when the settings it was made from change. Use it for anything annotators or other people build on.

Registration is explicit: import a new stage module in `pipeline/stages.py`, add the stage to a target in `TARGETS` if it should be runnable by a short name, and add new inputs to `configs/pipeline.yaml`. A non-manual stage is part of `all` automatically.

Hamilton versions a stage by the source of the stage function only. If the stage calls code whose changes should rerun it, make that visible in its inputs, as QA does by hashing its checks into their versions.

## Adding a QA check

Each check is one file, `qa/checks/<category>/<test>.py`, with a function registered by `@check(category, test, kind)`:

```python
@check("general", "constant_action", ACTION)
def constant_action(ctx, config: dict, spec: Check) -> dict:
    return run_category(ctx, config, spec)
```

The existing checks call their category's shared logic (`qa/action_qa.py`, `qa/video_checks.py`). A new check can instead implement its own logic: it receives the episode, with actions, states, profile, root attributes and videos decoded on first use, and returns its findings, skips and whether it applied.

To add a check, write the file, import it in `qa/checks/registry.py`, give it a section in `configs/action_qa_config.yaml` with `enabled`, `confidence` and `action`, and add it to the defaults in `qa/action_qa_config.py`. The next QA run runs only the new check. To remove a check, delete its import and file.

## Adding a converter

A converter is a module with `main(argv)` that writes `<episode>.h5` files in the Oopsie-Data format, with their videos, into `--output-dir`. Add it to `conversion/registry.py` under the name that `configs/imports.yaml` uses, with the flag it takes for its source ([imported datasets](pipeline/imports.md#converters)).

## Adding a training format

Write `training/writers/<format>.py` with `write(episodes_dir, out, included, layout, recipe)`, and add it to `training/writers/registry.py` and to `FORMATS` in `training/recipe.py`.

## Checking changes

There is no test suite. Before committing, lint the changed files and run the affected stage on a small mirror:

```bash
uvx ruff check --select F,E9 <files>
HF_TOKEN=... uv run oopsie ingest sync --data-dir ./data --smallest 10
uv run oopsie pipeline run <target> --set data_dir=./data --set index_path=./data/episode_index.parquet --set cache_dir=./data/cache
```

## Documentation

These docs are a [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) site built from `docs/`, with the navigation in `mkdocs.yml`:

```bash
uv run --group docs mkdocs serve     # preview at http://127.0.0.1:8000
uv run --group docs mkdocs build     # strict: fails on broken links
```

Every stage has a page under `docs/pipeline/`, which starts with what the stage is for and a table of its stage name, target, dependencies and outputs. A new stage gets a page and an entry in the navigation; new index columns and configs also go into the [reference](reference/index-columns.md).
