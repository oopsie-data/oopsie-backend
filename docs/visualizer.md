# Visualizer assets

Pipeline for the local visualizer and the public Hugging Face preview set ([Visualizer_Previews](https://huggingface.co/datasets/OopsieData-Submissions/Visualizer_Previews)).

```bash
cd oopsie-backend

# 1. Full local assets → oopsie-docs/docs/assets/visualizer/
#    Edit DATASET_ROOT / OUTPUT_ROOT in generate_visualizer_assets.py first.
#    Existing MP4s/posters are skipped; re-run rebuilds metadata.json.
uv run python -m oopsie_backend.scripts.generate_visualizer_assets

# 2. Face-safe public subset (400 failures + 100 successes, non-bulk only)
#    → oopsie-docs/docs/assets/visualizer-public/
#    Uses reports/face_candidate_{episodes,videos}.csv by default.
uv run python -m oopsie_backend.scripts.export_public_visualizer_subset

# 3. Replace HF dataset (requires HF_TOKEN with write access)
uv run python -m oopsie_backend.scripts.upload_public_visualizer_subset
```

Local/internal visualizer is unchanged by steps 2–3. Face CSVs and media trees stay gitignored.
