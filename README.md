# oopsie-backend

All oopsiedata processing code lives here

## Documentation

| Topic | Doc |
|-------|-----|
| Sync and stats (daily Slurm job) | [docs/sync-and-stats.md](docs/sync-and-stats.md) |
| HF org → local sync (`sync_hf_org`) | [docs/sync-hf-org.md](docs/sync-hf-org.md) |
| Dashboard stats (`compute_local_stats`, `compute_hf_stats`) | [docs/compute-stats.md](docs/compute-stats.md) |
| Final face-privacy pipeline and 2026-08-12 HF release | [docs/face-privacy-release-2026-08-12.md](docs/face-privacy-release-2026-08-12.md) |
| Private OopsieData v0.1 snapshot release | [docs/oopsiedata-v0.1-snapshot.md](docs/oopsiedata-v0.1-snapshot.md) |
| Qwen verification after Deface | [docs/qwen-face-verification.md](docs/qwen-face-verification.md) |
| Manual face and blur-policy annotation | [docs/face-blur-annotations.md](docs/face-blur-annotations.md) |
| Production tracked-Deface blurring | [docs/face-blur-pipeline.md](docs/face-blur-pipeline.md) |
| Deface benchmark and tracking details | [docs/deface-pipeline.md](docs/deface-pipeline.md) |
| Hugging Face face-presence scan | [docs/face-detection-scan.md](docs/face-detection-scan.md) |
| Preliminary detector-screening findings | [docs/face-detection-review-findings.md](docs/face-detection-review-findings.md) |
| Historical face-detector benchmark | [docs/face-blur-benchmark.md](docs/face-blur-benchmark.md) |
| Hand-labeled detector-development set | [docs/face-blur-testset.md](docs/face-blur-testset.md) |

## Recovering Original Unblurred Videos

The 240 videos replaced by the August 2026 face-privacy release are blurred on
the default Hugging Face `main` revisions. Their original, unblurred bytes are
preserved at the tag `pre-face-blur-2026-08-12` in each affected source dataset.
Use the same repository-relative path listed in that dataset's
`video_tags.csv`:

```bash
hf download OopsieData-Submissions/UofT_TISL \
  path/to/video.mp4 \
  --repo-type dataset \
  --revision pre-face-blur-2026-08-12
```

The tag is available to anyone who has read access to the corresponding source
dataset. A second copy of all 240 originals is retained in the private dataset
`OopsieData-Submissions/oopsie-face-raw-archive-2026-08-12`. See the
[release record](docs/face-privacy-release-2026-08-12.md#recovering-an-original-video)
for the repository list, exact source commits, and access implications.
