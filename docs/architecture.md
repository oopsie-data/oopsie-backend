# Architecture

The backend is built around two stores, the local mirror of the labs' data and the episode index, and a pipeline of stages that read from both and write their results to the index.

```mermaid
flowchart LR
  hub[("Hugging Face<br/>OopsieData-Submissions")] -->|sync| mirror[("local mirror<br/>h5 + videos")]
  imports["local imports"] -->|convert| mirror
  mirror --> stages["pipeline stages"]
  stages <--> index[("episode index")]
  stages --> outputs["reports, stats.json,<br/>eval split, annotation bundles,<br/>training sets"]
  annotators["annotators"] <-->|bundles and returns| stages
```

## The mirror

The mirror is a copy of every contributor and imported repo of the Hugging Face organization, kept on LS6 under `/corral/utexas/IRI26024/oopsiedata`. It is a superset of the Hub: the sync downloads new and changed files, and keeps files the Hub no longer has unless told to delete them. Datasets that are converted locally rather than uploaded live next to it, under `imports/`.

The h5 files in the mirror are the source of truth. No stage ever modifies them: labels, flags and corrections are stored in the index, and the annotated releases are written as separate copies.

## The episode index

`episode_index.parquet` is a table with one row per episode, keyed by `episode_key`, the first 16 hex characters of `sha1("<dataset_repo>/<repo_path>")`. The index build creates the rows and fills the core columns from the h5 files, and every later stage adds its results as its own columns: QA findings, duplicate links, the eval split, the collected annotations and so on. The [index column reference](reference/index-columns.md) lists them all.

Because every result is a column, any question about the data is a filter on one table:

```python
import polars as pl

df = pl.read_parquet("episode_index.parquet")
usable = df.filter(
    pl.col("on_disk") & pl.col("public") & pl.col("duplicate_of").is_null() & ~pl.col("critical").fill_null(False)
)
```

Stages write through `oopsie_backend.index.store.update_columns`, which replaces only the stage's own columns for the episodes it computed and leaves every other column untouched. Each write takes an exclusive lock on `<index>.lock` and replaces the file atomically, so stages that run at the same time never lose each other's results.

## Repos

The index build discovers episodes under `<data_dir>/OopsieData-Submissions/<repo>/` and `<data_dir>/imports/<name>/`. Each repo has a kind that decides how the rest of the pipeline treats it:

| Kind | Repos | Treatment |
|---|---|---|
| `contributor` | every org repo not listed in a config | indexed, QA-checked and counted; public unless the lab id is a placeholder |
| `external` | the entries of `configs/imports.yaml` | indexed, QA-checked and usable for training, but never public and never counted toward the lab requirements |
| `archive` | frozen snapshots listed in `configs/repositories.yaml` | never synced or indexed |
| `infra` | site assets and the annotation repos, listed in `configs/repositories.yaml` | never synced or indexed |

## Which episodes count

Three columns take episodes out of later steps, each for a different reason:

- **`duplicate_of`**: the episode is a copy of another one. Every downstream step skips copies, so each recording counts once.
- **`critical`**: the episode is structurally unusable, for example unreadable or with misaligned actions. The stats, split planning and training leave it out.
- **`annotation_excluded`**: the annotators reported a problem that makes the episode unusable, such as a cut-off recording. Training and the dataset analysis leave it out.

## Guarantees

The eval split must never change once annotators have started on it. Three rules ensure this:

- The chosen eval proposal, the annotated train sample and the annotator assignments are **written once**. A run records the settings each was made from, and fails rather than rewrite it when those settings change.
- Once a proposal is chosen, the proposal stage stops regenerating proposals, so changes to QA or to the critical conditions cannot alter them.
- Training keeps a held-out eval episode even if it becomes critical later. The only way an episode leaves the eval set is through the annotators' flags.

## The pipeline

Stages are Python functions registered with `@stage` and run by [Apache Hamilton](https://hamilton.apache.org/), which reruns a stage only when its code or inputs changed. The [pipeline overview](pipeline/index.md) describes the stages, how to run them and how the cache decides what to rerun.
