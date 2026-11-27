# Running targets

`oopsie pipeline run <target>` runs a target's stages, after first running any stage they depend on whose result is out of date. This is how the nightly job runs, and the normal way to run anything in the pipeline. The stages themselves are listed in the [overview](index.md#stages).

## Running a target

```bash
uv run oopsie pipeline run all --config configs/pipeline.yaml --explain   # the nightly run
uv run oopsie pipeline run qa                                             # one target and what it needs
uv run oopsie pipeline run --list                                         # targets and stages
```

Running a target also runs every stage it depends on, unless that stage's result is cached. `--explain` prints which stages ran and which came from the cache. Other options:

| Option | Effect |
|---|---|
| `--config FILE` | the inputs file, by default `configs/pipeline.yaml` |
| `--set KEY=VALUE` | overrides one input, e.g. `--set data_dir=/path/to/mirror` or `--set use_hub=false`; repeatable |
| `--no-cache` | reruns every requested stage |
| `--serial` | runs one stage at a time |
| `--graph FILE` | draws the stages of the target to an image |

To run one stage's work on its own, with your own options, see [Running commands](running-commands.md).

## Targets

| Target | Stages |
|---|---|
| `all` | every stage except the manual ones; the nightly job runs this |
| `sync` | `hub_mirror` |
| `index` | `episode_index`, `hub_upload_times` |
| `preqa` | `duplicate_episodes`, `duplicate_videos`, `unified_camera_names`, `unified_embodiments`, `unified_policies` |
| `qa` | `qa_results` |
| `postqa` | `critical_episodes` |
| `stats` | `stats` |
| `proposals` | `eval_proposals` |
| `splits` | `eval_split`, `train_sample` |
| `annotation` | `annotation_flags`, `annotation_publish` |
| `imports` | `imported_datasets` (manual) |
| `training` | `training_data` (manual); needs `--set training_recipe=configs/training/<recipe>.yaml` |
| `analysis` | `dataset_analysis`, `annotation_analysis` (manual); needs `uv sync --extra analysis` |
| `release` | `data_release` (manual); needs `--set release_snapshot=<sync report>` |

Manual stages are left out of `all` because they are expensive or need a decision: which dataset to convert, which recipe to build, when to publish a release.

## Inputs

`configs/pipeline.yaml` holds every input a stage may take. The defaults are the LS6 paths.

| Input | Meaning |
|---|---|
| `data_dir` | the mirror root |
| `index_path` | the episode index |
| `cache_dir` | Hamilton's cache |
| `workers` | parallel processes of the index build, QA and training |
| `use_hub` | whether to sync with the Hub and list Hub hashes; `false` works offline |
| `reports_dir` | where reports go |
| `stats_docs_output` | the website's `stats.json` |
| `split_config_path`, `annotation_config_path` | the eval split and annotation configs |
| `training_dir`, `training_work_dir`, `training_recipe` | where training sets are built and copied, and which recipe |
| `stage_dir` | fast storage where imported datasets are converted |
| `node_stage_dir` | node-local storage that QA and training copy episodes to before reading them, at most 1,000 at a time |
| `release_snapshot`, `release_name` | the sync report a release is made from, and its name |

## What reruns

Hamilton keeps, for each stage, a version of the stage's code and a fingerprint of each of its inputs. When both match an earlier run, it returns the stored result instead of running the stage. A stage's result is itself a fingerprint of what it wrote, usually a hash of its index columns or output files, so a stage that reruns but produces the same output does not trigger the stages after it.

Hamilton cannot see files on its own. Small stages that run on every invocation (`@stage(always_run=True)`) therefore watch the outside state and return its hash:

| Watcher | Hashes |
|---|---|
| `mirror_state` | the path, size and modification time of every h5 file, and the repo and import configs |
| `camera_names_config`, `embodiments_config`, `policies_config`, `critical_config`, `stats_config`, `split_config`, `robot_aliases_config`, `annotation_flags_config`, `analysis_config` | their config file |
| `qa_check_versions` | the version of every QA check |
| `annotation_roster` | `annotators.txt`, `train_annotators.txt` and the bundle roster |
| `annotation_state` | the annotation stores |
| `training_recipe_state` | the training recipe |

When a watched file changes, the stage after the watcher sees a new input and reruns. `hub_mirror` and `imported_datasets` also run every time, because only they can find out whether there is new data.

Hamilton versions a stage by the source of the stage function alone, not by the code it calls. Changes to that code are usually caught anyway: the index stages see changed files, and QA hashes its checks' code into their versions. After other code changes, run the affected target once with `--no-cache`.

Two stages also wait for a person's decision before they run: see [human decisions and guarantees](index.md#human-decisions-and-guarantees).
