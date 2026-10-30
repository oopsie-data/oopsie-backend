# Annotation coordinator runbook

`batch_1` holds the eval assignment and `batch_2` the train sample, as set in `configs/annotation_pipeline.yaml`. Annotators only see the batch names.

The method behind the splits and assignments is in [eval-split.md](eval-split.md), and the annotator instructions are in [annotator-guide.md](annotator-guide.md).

## Keep these files

All annotation files live in `/corral/utexas/IRI26024/oopsiedata/annotation/`, next to the mirror's `OopsieData-Submissions/` folder, so every run on LS6 sees them. Episode discovery only reads the org folder, so these copies are never indexed as data.

| File in `annotation/` | Why |
|---|---|
| `eval_assignment/`, `train_assignment/` | The only mapping from hashed episode ids back to source episodes |
| `eval_annotations.jsonl`, `train_annotations.jsonl` | Append-only stores of every collected annotation |
| `returns/` | Downloaded pull requests; `collect` can be re-run from them |
| `annotation_repo/` | The staged Hugging Face repo |
| `eval_release/`, `train_release/` | Source episodes with collected annotations |

`configs/annotation_pipeline.yaml` holds the batch repo (`repo_id`), the returns repo (`returns_repo_id`) and these paths, relative to `<data_dir>/annotation`. Both repos are `infra` repos in `configs/repositories.yaml`, so the Hub sync never downloads them.

Do not regenerate the assignments once the repo is published. A different assignment changes which episode sits at which position.

## Commands

```bash
P="uv run python -m oopsie_backend.annotation.pipeline --config configs/annotation_pipeline.yaml"
```

| Step | Where | Command |
|---|---|---|
| Package both batches | LS6 | `$P stage` |
| Create private repos and upload | LS6 | `$P publish` |
| Download returned pull requests | anywhere | `$P pull` |
| Add returns to the stores | anywhere | `$P collect` |
| Progress per annotator | anywhere | `$P status` |
| Build annotated release copies | LS6 | `$P release` |

Every command reads and writes the files in `annotation/`. To work on a copy elsewhere, pass `--data-dir <mirror root>` (the folder holding `annotation/`) and `--index <episode index>`.

## 1. Before distribution

1. Write the assignments to `/corral/utexas/IRI26024/oopsiedata/annotation/eval_assignment/` and `train_assignment/` (the default output of `oopsie annotation assign` is `eval_assignment/` there).
2. Give every annotator's Hugging Face account read access to both repos.

## 2. Publish (LS6)

```bash
$P stage
$P publish
```

`stage` exits non-zero if an episode lacks its h5 or videos, and still packages the rest. It writes `annotation_repo/` with one folder per annotator and a `README.md` that lists annotators, folders and episode counts. `publish` creates the batch repo `OopsieData-Submissions/oopsie-annotation` and the empty returns repo `OopsieData-Submissions/oopsie-annotation-returns` as private datasets, and uploads the folder to the batch repo. It refuses to upload if either repo already exists and is public.

Check on Hugging Face that the repo is private and the README table is complete, then send annotators the [annotator guide](annotator-guide.md).

## 3. During annotation

Run this loop as returns arrive:

```bash
$P pull       # add --close to close collected pull requests
$P collect
$P status
```

Annotators download their folder with `oopsie-data annotation download` and return annotated files with `oopsie-data annotation upload`, which opens a pull request on the returns repo. `pull` downloads accepted pull requests, with the folder's manifests from the batch repo, to `returns/pr_NNNN/` and skips those it already has. A pull request is rejected if it touches more than one annotator folder or changes anything other than episode `.h5` files. Rejected pull requests appear in `status` and in `returns/pull_requests.json`; ask the annotator to upload again with `oopsie-data annotation upload`.

Pull requests never need to be merged. When an annotator uploads twice, both versions are stored and the later annotation counts.

`status` prints returned/assigned annotations per annotator and batch. Per-batch details, such as missing episodes per annotator and outcome agreement on eval, are in `eval_annotations_summary.json` and `train_annotations_summary.json`.

## 4. Release (LS6)

Copy the stores to LS6, then run:

```bash
$P release
```

This writes `eval_release/` and `train_release/`: copies of each annotated source episode with its videos, with every collected annotation added as its own `episode_annotations` group. The lab's own annotations are kept. Re-running after more returns updates only the changed groups.

## Checking a new repo

Before announcing new repos, download one folder with `oopsie-data annotation download`, annotate an episode, upload it with `oopsie-data annotation upload`, then run `pull` and `collect`. `pull` reads the changed files from the pull request diff, so this confirms that returns are accepted.
