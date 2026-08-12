# Visualizer assets

Pipeline for the local visualizer and the public Hugging Face preview set ([Visualizer_Previews](https://huggingface.co/datasets/OopsieData-Submissions/Visualizer_Previews)).

```bash
cd oopsie-backend

# 1. Full local assets → oopsie-docs/docs/assets/visualizer/
#    Edit DATASET_ROOT / OUTPUT_ROOT in generate_visualizer_assets.py first.
#    Existing MP4s/posters are skipped; re-run rebuilds metadata.json.
uv run python -m oopsie_backend.scripts.generate_visualizer_assets

# 2. Face-safe, dataset-balanced public subset
#    (400 failures + 100 successes, non-bulk only)
#    → oopsie-docs/docs/assets/visualizer-public/
#    Uses reports/face_candidate_{episodes,videos}.csv by default.
uv run python -m oopsie_backend.scripts.export_public_visualizer_subset

# 3. Replace HF dataset (requires HF_TOKEN with write access)
uv run python -m oopsie_backend.scripts.upload_public_visualizer_subset
```

Local/internal visualizer is unchanged by steps 2–3. Face CSVs and media trees stay gitignored.

## Dataset-balanced sampling

The public exporter groups episodes by dataset before sampling. For generated
metadata, the dataset is the leaf of `dataset_repo`; remote metadata may instead
provide `dataset_name` or a first-level folder in `hf_path`/`h5_path`.

The default selection reserves up to 10 failures and 3 successes per dataset,
then distributes remaining slots using the square root of each dataset's
available episode count. This gives small datasets a floor without allowing the
largest datasets to dominate. Datasets with fewer episodes than the floor
contribute all available episodes.

```bash
uv run python -m oopsie_backend.scripts.export_public_visualizer_subset \
  --min-failure-per-dataset 10 \
  --min-success-per-dataset 3 \
  --dataset-weight-power 0.5
```

Set `--dataset-weight-power 0` for near-equal allocation or `1` for proportional
allocation. Named quotas are printed locally, but dataset identities remain
excluded from the public `metadata.json`.

For roughly equal total episode counts per dataset, allocate total quotas before
splitting them into failures and successes:

```bash
uv run python -m oopsie_backend.scripts.export_public_visualizer_subset \
  --sampling-strategy equal-total
```

This capacity-bounded strategy gives small datasets all their available
episodes, distributes the remaining total slots nearly equally, and then fits
the exact requested failure/success totals within those dataset quotas.

## Direct Hugging Face export

To build the public subset without downloading the complete source dataset, run:

```bash
uv run python -m oopsie_backend.scripts.export_public_visualizer_subset_from_hf \
  --repo-id zhouzypaul/oopsie-data \
  --output-root assets/visualizer-public-hf \
  --cache-dir assets/hf-cache \
  --metadata-workers 1 \
  --workers 4
```

This caches eligible HDF5 metadata, applies the same face filtering and
dataset-balanced 400-failure/100-success selection, and downloads only the one
selected camera video per episode. The source revision and sampling settings are
stored in the scrubbed output metadata.

If a download or transcode is interrupted, rerun the same command with
`--resume`; completed videos and posters are reused.

For `CMU_Intent`, tactile streams are removed before sampling and `front` is
preferred whenever it is face-safe and available. Episodes without front use a
non-tactile visual fallback. CMU episodes left with only a tactile stream after
face filtering are excluded from the candidate pool.
