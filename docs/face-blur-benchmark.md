# Face Detector Benchmark

The benchmark runs every frame of the hand-labeled face-blur video set through
one or more detector-only models. It reports video-level recall, specificity,
precision, uncertain-video behavior, false detections and raw box area on
negative frames, and exact-node throughput.

This page records detector-development experiments. The finalized August 2026
release used Deface CenterFace for screening and tracked rendering, with Qwen
verification and manual blur decisions between those stages. SCRFD benchmark
results below are historical comparisons, not the production release policy.
See [`face-privacy-release-2026-08-12.md`](face-privacy-release-2026-08-12.md).

It does **not** report frame-level privacy coverage because the current labels
only say whether an entire video contains face-like content. See
[`face-blur-testset.md`](face-blur-testset.md) for the label semantics and known
task-relevant printed-face cases.

## Detectors

The pinned registry contains:

| Name | Model source | Runtime |
|---|---|---|
| `yunet` | OpenCV Zoo YuNet 2023mar | OpenCV DNN CPU |
| `scrfd-500m` | InsightFace v0.7 `buffalo_sc` detector member | ONNX Runtime |
| `scrfd-2.5g` | InsightFace v0.7 `buffalo_m` detector member | ONNX Runtime |
| `scrfd-10g` | InsightFace v0.7 `buffalo_l` detector member | ONNX Runtime |
| `scrfd-34g` | InsightFace SCRFD-34G pretrained detector converted to ONNX by `RuteNL/SCRFD-face-detection-ONNX` | ONNX Runtime |
| `deface-centerface` | CenterFace model bundled with ORB-HD/deface v1.5.0 | deface + ONNX Runtime |

Only the detector ONNX member is extracted from InsightFace archives. Face
recognition, demographic attribute, and alignment models are not retained or
loaded. Every archive and extracted model has a pinned SHA-256 checksum in
`oopsie_backend/face_blur/models.py`.

InsightFace code is MIT licensed; its published model weights are available for
non-commercial research. YuNet's model directory is MIT licensed.
The adapted InsightFace decoder notice is retained in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Install

CPU-only:

```bash
uv sync --extra face-blur
```

GPU, including PyPI CUDA and cuDNN runtime libraries:

```bash
uv sync --extra face-blur-gpu
```

Do not install both extras in the same environment because both distributions
provide the `onnxruntime` Python module.

## Download Models

The benchmark downloads missing models automatically. They can also be
provisioned in advance:

```bash
uv run python -m oopsie_backend.scripts.download_face_detector_models \
  --model-dir face_blur_models
```

Set `OOPSIE_FACE_MODEL_DIR` to change the default outside the CLI.

## Run

Run all models on every frame with GPU SCRFD inference:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.evaluate_face_detectors \
  --manifest face_blur_testset/manifest.csv \
  --model-dir face_blur_models \
  --output-dir face_blur_benchmarks/latest \
  --device cuda:0
```

Smoke-test the first two videos on CPU:

```bash
uv run --extra face-blur \
  python -m oopsie_backend.scripts.evaluate_face_detectors \
  --detectors yunet scrfd-500m \
  --limit 2 \
  --max-frames 10 \
  --device cpu
```

Important options:

| Option | Default | Meaning |
|---|---:|---|
| `--input-size` | `640x640` | Letterboxed detector input; dimensions must be multiples of 32 |
| `--inference-threshold` | `0.1` | Low score floor retained in raw results |
| `--thresholds` | `0.2 ... 0.7` | Operating points summarized without rerunning inference |
| `--primary-threshold` | `0.5` | Operating point used for the main report and review images |
| `--no-resume` | off | Recompute even when source and detector configuration match |
| `--no-review-images` | off | Skip internal annotated JPEGs |

The inference floor must be no greater than every reported operating point.

## Artifacts

```text
face_blur_benchmarks/latest/
  comparison.json
  comparison.csv
  <detector>/
    config.json
    summary.json
    thresholds.csv
    videos.csv
    detections/<sample_id>.json.gz
    review/<label>/<sample_id>__review.jpg
```

Per-video results are atomic, compressed, and resumable. Each successful result
records the source SHA-256, model/configuration hash, video metadata, timings,
and sparse per-frame detections. Failures are written explicitly and cause a
nonzero command exit.

Review images may contain faces and are intentionally ignored by git. Treat the
whole benchmark output as internal data.

The `deface-centerface` adapter runs at each frame's native resolution, matching
deface's default behavior. The general `--input-size` option applies to SCRFD
and YuNet but is intentionally ignored for this detector.

When detectors are launched as separate parallel commands, combine their
finished summaries without rerunning inference:

```bash
uv run python -m oopsie_backend.scripts.compare_face_detector_benchmarks \
  --output-dir face_blur_benchmarks/latest
```

The `uncertain` videos are reported separately and never counted as positives
or negatives. A detector selected using this corpus must still be evaluated on
a separately labeled frame/track-level holdout before the blur pipeline is
treated as privacy-ready.

## Initial Baseline: 2026-07-15

All four detectors were run at `640x640` over all 65,356 frames in the 101-video
test set. The table uses threshold `0.3`, the highest tested operating point at
which every detector found all 30 positive videos.

| Detector | Positive videos | Negative videos triggered | Negative frames triggered | Raw box area on negative frames | Pipeline FPS |
|---|---:|---:|---:|---:|---:|
| YuNet | 30/30 | 53/58 | 14.223% | 0.5198% | 86.8 |
| SCRFD-500M | 30/30 | 56/58 | 10.989% | 0.3720% | 169.5 |
| SCRFD-2.5G | 30/30 | 55/58 | 7.933% | 0.5446% | 85.6 |
| SCRFD-10G | 30/30 | 40/58 | 1.960% | 0.0343% | 77.5 |

These throughput figures came from concurrent jobs rather than isolated
single-model timing. SCRFD-10G's first 20 videos also ran on a GPU contended by
an unrelated workload before the resumable run moved to a free GPU. Treat the
figures as operational lower bounds, not clean model latency measurements.

At the time of this baseline, SCRFD-10G at threshold `0.3` was selected as the
default experimental renderer because it retained full video-level recall while
reducing negative-frame triggers by roughly 4x versus SCRFD-2.5G and 7x versus
YuNet. That selection was later superseded by the finalized tracked-Deface
workflow. The baseline was never a production privacy threshold: 40 negative
videos still contained at least one false detection, and video-level labels
cannot measure exposed face frames or mask coverage.

## SCRFD-34G Follow-up: 2026-07-20

SCRFD-34G was added and benchmarked at `640x640` over the same 101-video set.
It did not improve false positives on this robot-manipulation corpus. At
threshold `0.4`, it retained 30/30 positive videos but still triggered 51/58
negative videos, 22.213% of negative frames, and 5.7428% raw negative-frame box
area. Its detector-only pipeline FPS was 50.8 in this run.

For comparison, SCRFD-10G at threshold `0.3` triggered 40/58 negative videos,
1.960% of negative frames, and 0.0343% raw negative-frame box area while also
retaining 30/30 positive videos. The temporal blur planner amplifies this gap:
with default hold/interpolation settings, SCRFD-34G at `0.4` would mask 35.044%
of negative frames, while SCRFD-10G at `0.3` would mask 6.708%.

## Deface CenterFace Follow-up: 2026-07-24

The native-resolution CenterFace detector from deface v1.5.0 was benchmarked
over the same 101 videos and 65,356 frames. Native threshold `0.2` detected all
30 positive videos, triggered 54/58 negative videos and 7.327% of negative
frames, and covered 0.02313% raw negative-frame box area. At threshold `0.3`,
it still detected 30/30 positives while triggering 30/58 negative videos,
2.099% of negative frames, and 0.00683% raw negative-frame box area. Pipeline
throughput was 149.8 FPS on CUDA.

See [the deface pipeline report](deface-pipeline.md) for the full comparison,
native and tracked review-video commands, and the observed frame-level temporal
tradeoff. The confirmed tracked policy retains 30/30 positive videos and 30/58
triggered negative videos while masking 6.159% of negative frames, versus
6.708% for the default SCRFD-10G temporal plan.

## Production Selection: August 2026

The benchmark motivated the use of Deface CenterFace, but the released workflow
did not select videos from the 101-video benchmark alone. It first used Deface
0.3 to triage repositories, then rescanned all 4,086 videos in the five
prioritized repositories at threshold `0.4`. That scan produced 1,370
candidates. Pinned Qwen verification reduced the primary review pool to 257,
and a broader 452-video manual queue—including strong model disagreements—set
the final 240 blur actions.

Rendering used tracked Deface with `0.3` strong seeds and `0.2` continuations,
not the `0.4` screening threshold. Ninety-seven suspect renders then received a
stricter fallback correction before the final files were promoted to Hugging
Face `main`. These production counts and controls supersede any earlier wording
that calls an SCRFD operating point the default release pipeline.
