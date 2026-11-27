# Train sample

Besides the eval set, annotators label a sample of training episodes. Their labels add independent failure annotations to training episodes.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `train_sample` | `splits` | `chosen_proposal` | `<data_dir>/annotation/train_annotation_sample.json`, written once |

## Sampling

The sample is drawn from the chosen proposal's train set: every eligible episode that is not held out, not `excluded`, and not from a held-out lab or task. Its size is `train_sample.num_episodes` in `configs/eval_split.yaml`.

The sample aims for coverage rather than proportions. The budget is split equally over labs, and within each lab over scenes, tasks, policies and success labels. A lab with fewer episodes than its share passes the rest to the other labs.

The output has the layout of a proposal with a single split, `train_annotated`, so the [first assignment](annotation/index.md#bundles-and-the-roster) treats it like the eval set. Like the chosen proposal, it is written once and kept as long as the proposal and settings it was made from are unchanged. Annotators who join later each get a new train bundle drawn from the same pool, which grows the annotated train set beyond this sample ([train bundles](annotation/index.md#train-bundles)).

## Running it by hand

```bash
uv run oopsie splits sample-train --index episode_index.parquet \
  --proposal eval_split_proposals/proposal_02.json --num-episodes 1000
```
