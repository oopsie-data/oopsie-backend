# Reading the index

Every release ships `episode_index.parquet`, a table with one row per episode. Its columns describe the episode (lab, robot, task, cameras, length) and hold the results of the team's processing: duplicate links, quality checks, the eval split and the labels. Any question about the data is a filter on this table.

```python
import polars as pl

df = pl.read_parquet("/path/to/data/releases/2026-11-08/episode_index.parquet")
```

## Browse as HTML

Create a standalone viewer and open the resulting HTML file in a browser.

```bash
uv run oopsie tools parquet-html data/releases/2026-11-08/episode_index_sample.parquet
```

The HTML is written beside the source with the same name and an `.html` extension.
It works offline and provides search, sorting, pagination, visible-column controls,
column types and a row inspector for nested lists, annotations and QA results.
The viewer displays index metadata and leaves the source Parquet file intact.

By default it embeds the first 1,000 rows and labels the preview with the full row
count. Search and sorting operate on embedded rows only. Use `--max-rows 0` to
embed all rows (large indexes can produce large HTML files), `--columns NAME ...`
to export selected columns, or `--out PATH.html` to choose the output path.

```bash
uv run oopsie tools parquet-html data/releases/2026-11-08/episode_index.parquet \
  --max-rows 0 --columns episode_key dataset_repo robot_name language_instruction \
  --out reports/episode_index.html
```

Timestamps use ISO strings. Integers beyond JavaScript's exact integer range,
decimals and nonfinite numbers are displayed as strings to preserve their values.

## Key columns

| Column | Content |
|---|---|
| `episode_key` | the row key, unique per episode |
| `dataset_repo`, `repo_path` | the lab's repo and the h5 path inside it |
| `videos` | one entry per camera, with `camera` and the mp4's `path` inside the repo |
| `repo_kind` | `contributor` for data uploaded by a participating lab, `external` for converted external datasets |
| `lab_id`, `robot_name`, `gripper_name` | who recorded the episode and with which hardware |
| `language_instruction`, `task` | the instruction as recorded, and normalized (lowercase, collapsed whitespace) |
| `cameras`, `num_steps`, `duration_seconds`, `control_freq` | the recorded cameras and the episode's length |
| `annotations` | the lab's own labels: success, outcome, failure categories, severity, description |
| `eval_split` | which eval split holds the episode out, see [below](#test-and-train); null for train episodes |
| `duplicate_of` | set when the episode is a copy of another one |
| `critical` | the episode is structurally unusable, e.g. unreadable or with misaligned actions |
| `annotation_excluded` | annotators reported a problem that makes the episode unusable |

The [index column reference](../reference/index-columns.md) lists every column.

## Usable episodes

```python
usable = df.filter(
    pl.col("duplicate_of").is_null()
    & ~pl.col("critical").fill_null(False)
    & ~pl.col("annotation_excluded").fill_null(False)
)
```

## Test and train

`eval_split` holds out episodes for evaluation:

| `eval_split` | Meaning |
|---|---|
| `id` | test: scene and task both also appear in train |
| `od_task_id_scene` | test: a task never seen in train, in a scene that is |
| `od_task_od_scene` | a lab never seen in train; empty in the current releases |
| `excluded` | held-out labs and tasks beyond the annotation budget; neither train nor test |
| null | train |

```python
test = df.filter(pl.col("eval_split").is_in(["id", "od_task_id_scene"]))
train = df.filter(pl.col("eval_split").is_null())
```

Never train on `excluded` episodes: they come from the same held-out labs and tasks as the test set. The [eval split](../pipeline/eval-split.md) page explains how the split is chosen.

## From a row to its files

An episode's files are in its repo folder under the data folder:

```python
from pathlib import Path

data = Path("/path/to/data")
row = df.row(0, named=True)
h5 = data / row["dataset_repo"] / row["repo_path"]
mp4s = {v["camera"]: data / row["dataset_repo"] / v["path"] for v in row["videos"]}
```

`oopsie tools inspect-hdf5 <file>` prints the tree of an h5 file.

## Labels

The `annotations` column holds the labels each lab recorded with its own episodes; they are also stored inside the h5 files.

<!-- TODO: describe the human annotations once it is decided whether releases for outside users include them. -->
