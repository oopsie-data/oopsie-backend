# Deface Detector and Tracked Renderer

The finalized August 2026 face-privacy release used the CenterFace model from
[ORB-HD/deface](https://github.com/ORB-HD/deface) v1.5.0 for both candidate
screening and tracked rendering. The native and SCRFD modes remain useful
comparisons, but they did not produce the 240 derivatives published to Hugging
Face `main`.

See [`face-privacy-release-2026-08-12.md`](face-privacy-release-2026-08-12.md)
for the release outcome and [`face-blur-pipeline.md`](face-blur-pipeline.md)
for the production rendering and correction policy.

Deface uses CenterFace independently on every frame. Its native defaults are:

- detection at the source frame's native resolution;
- score threshold `0.2`;
- no temporal tracking, interpolation, pre-roll, or hold;
- a `1.3` mask scale;
- ellipse-shaped blur masks.

The batch wrapper preserves those visual defaults. It copies audio by default
to make review files comparable with the existing pipeline, whereas deface's
standalone CLI drops audio unless requested. Outputs are written through a
temporary path, validated, atomically published, and accompanied by a
`<video>.deface.json` audit sidecar.

## Install

Use either face-blur extra. Both include deface 1.5.0 and ONNX;
the GPU extra uses the same ONNX Runtime CUDA installation as SCRFD.

```bash
uv sync --extra face-blur
uv sync --extra face-blur-gpu
```

## Historical Detector Benchmark

The `deface-centerface` adapter exposes CenterFace boxes and scores through the
same benchmark used for SCRFD and YuNet. Run inference at a low score floor so
multiple operating thresholds can be summarized without rerunning the model:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.evaluate_face_detectors \
  --manifest face_blur_testset/manifest.csv \
  --detectors deface-centerface \
  --inference-threshold 0.1 \
  --thresholds 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
  --primary-threshold 0.2 \
  --device cuda:0 \
  --output-dir face_blur_benchmarks/deface-native
```

Although the benchmark accepts the common `--input-size` option, CenterFace
intentionally ignores it and uses native frame resolution, matching deface.

### Same-set results: 2026-07-24

CenterFace was evaluated on the same 101 videos and all 65,356 frames used for
the SCRFD benchmark. There were no failed videos. The native deface threshold
is `0.2`; `0.3` is also shown because it is the highest tested threshold that
retained 30/30 positive-video recall and gives a more direct operating-point
comparison with SCRFD-10G.

| Detector / threshold | Positive videos | Negative videos triggered | Negative frames triggered | Raw box area on negative frames | Precision | Uncertain videos detected | Pipeline FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deface CenterFace / `0.2` (native) | 30/30 | 54/58 | 2,793/38,121 (7.327%) | 0.02313% | 0.357 | 13/13 | 149.8 |
| Deface CenterFace / `0.3` | 30/30 | 30/58 | 800/38,121 (2.099%) | 0.00683% | 0.500 | 12/13 | 149.8 |
| SCRFD-10G / `0.3` | 30/30 | 40/58 | 747/38,121 (1.960%) | 0.03432% | 0.429 | 12/13 | 77.5 |

Deface at `0.3` triggers fewer negative videos and covers less raw negative box
area than SCRFD-10G, but triggers slightly more negative frames. The throughput
figures are operational measurements, not a clean latency comparison: the
older SCRFD run was partially GPU-contended.

A frame check of `video_0017` exposes the more important temporal tradeoff. At
frame 0, CenterFace scores the visible profile at `0.2683`, so native deface
blurs it while SCRFD-10G at `0.3` does not. At frame 4, CenterFace misses that
face (its only above-threshold result is an unrelated tiny box scored `0.2070`),
while SCRFD detects the face at `0.3444` and the temporal pipeline masks it.
Native deface can therefore close an entry-frame leak yet still flicker or
leave later frames exposed because detections are not linked across time.

Complete machine-readable metrics and per-frame detections are in
`face_blur_benchmarks/deface-native/`.

## Historical Positive-Set Review Renders

For the aggregate comparison operating point, use threshold `0.3`. It retains
30/30 positive-video recall while improving precision and negative-video
specificity over the native `0.2` default:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.run_deface \
  --manifest face_blur_testset/manifest.csv \
  --labels yes \
  --threshold 0.3 \
  --output-dir face_blur_outputs/deface-positive-review-t030 \
  --device cuda:0
```

Review videos are placed under
`face_blur_outputs/deface-positive-review-t030/videos/`, preserving the
manifest's relative paths. The batch report is `deface-run.json`. The completed
comparison set contains all 30 positive videos (21,717 frames), 30 audit
sidecars, and no validation failures. Every output was checked for source
dimensions, frame count, frame rate, and recorded SHA-256. The writer disables
imageio's default 16-pixel macroblock padding so source dimensions are
preserved exactly.

The native-default `0.2` render is retained separately under
`face_blur_outputs/deface-positive-review/`. It is useful for privacy-oriented
review because it catches some low-score face frames that `0.3` drops. In
particular, the visible frame-0 profile in `video_0017` scores `0.2683`. Neither
threshold should be selected from video-level recall alone; that metric only
requires one detection somewhere in a positive video and cannot measure
per-frame face coverage.

For a controlled visual comparison, open each file alongside the matching
video under `face_blur_outputs/positive-review/videos/`. Native deface masks
only frames with a direct CenterFace detection; the tracked mode below can also
mask interpolated, pre-roll, and held frames.

## Confirmed Temporal Tracking

The tracked-deface mode retains the `0.3` operating point without
allowing every weak or isolated detection to create a persistent blur:

1. CenterFace runs at native resolution with a `0.2` continuation floor.
2. A track is confirmed only after two spatially consistent detections score at
   least `0.3`.
3. Once confirmed, matching `0.2`–`0.3` observations may extend the track in
   either direction because planning happens before rendering.
4. Gaps of at most eight frames are interpolated, with two frames of pre-roll
   and two frames of hold.
5. An unconfirmed `0.3` detection is blurred on its original frame only. It is
   never extended temporally; unseeded low-score tracks are discarded.
6. Planned boxes use deface's native `1.3` ellipse anonymizer rather than the
   SCRFD pipeline's padded rectangular blur.

Render the positive comparison set:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.run_tracked_deface \
  --manifest face_blur_testset/manifest.csv \
  --labels yes \
  --output-dir face_blur_outputs/tracked-deface-positive-review \
  --device cuda:0
```

The completed set contains all 30 positive videos and 21,717 frames. Its 30
MP4 files and 30 `<video>.tracked-deface.json` sidecars passed dimension, frame
count, frame-rate, configuration, and output-hash validation. The batch report
is `tracked-deface-run.json`.

The policy can be evaluated without rerunning CenterFace because the detector
benchmark retained results down to `0.1`:

```bash
uv run python -m oopsie_backend.scripts.evaluate_tracked_deface \
  --detections-dir \
    face_blur_benchmarks/deface-native/deface-centerface/detections \
  --output face_blur_benchmarks/deface-native/tracked-deface/summary.json
```

### Tracked results

| Pipeline | Positive videos | Negative videos triggered | Negative frames masked | Precision |
|---|---:|---:|---:|---:|
| Native deface `0.3` | 30/30 | 30/58 | 800/38,121 (2.099%) | 0.500 |
| Confirmed tracked deface | 30/30 | 30/58 | 2,348/38,121 (6.159%) | 0.500 |
| SCRFD-10G temporal | 30/30 | 40/58 | 6.708% | 0.429 |

Tracking does not add any newly triggered negative videos, but necessarily
extends masks within videos already triggered by CenterFace. Its planned raw
box footprint is 0.02263% of all negative-video pixels; the estimated scaled
ellipse footprint is 0.02988%. Overlapping masks are counted separately in
that estimate.

Across positive videos, 510 low-score candidate tracks were discarded, 207
tracks were confirmed, and 22 isolated strong detections were rendered only on
their source frame. `video_0017` now has a continuous face mask through frames
0–12, covering the entry frame missed by deface `0.3` and frame 4 missed by
native deface `0.2`.

## Final Production Use

The release used two deliberately different CenterFace thresholds for two
different jobs:

- **Screening:** threshold `0.4` and NMS threshold `0.4` identified 1,370
  candidates among 4,086 videos in the five prioritized repositories. This
  stage stopped after a video's first positive frame.
- **Rendering:** the full video was decoded at native resolution with a `0.2`
  continuation floor. Two spatially consistent detections at `0.3` confirmed a
  track; gaps of up to eight frames were interpolated with two-frame pre-roll
  and two-frame hold.

The screening threshold was not reused as the render threshold: a selective
video-level candidate pass and conservative frame-level coverage solve
different problems. Qwen then filtered candidate frames semantically, but the
manual `blur_action` label—not either model—selected the final 240 inputs.

After review of the initial outputs, 97 suspect videos were rerendered with a
stricter fallback policy. Confirmed-track settings remained the same; fallback
tracks received two-frame pre-roll and 15-frame hold. All 97 corrections passed
machine validation, 14 were manually approved as spot checks, and the corrected
set was included before promotion to `main`.

The released media retained the deface-style ellipse at scale `1.3`, Gaussian
blur, H.264 CRF 18 with the `medium` preset, and copied audio. Complete hashes
and configuration records are stored in each affected dataset under
`face_blur_release/2026-08-12/`.

## Important Limitation

The development corpus has video-level labels, so the benchmark can report the same
video-level recall and negative-video/frame trigger metrics as SCRFD. It still
cannot measure the fraction of actual face frames covered or whether an ellipse
fully obscures identity. The eight-frame association window also represents
different durations at different video frame rates. The finalized release
therefore added Qwen-assisted candidate review, explicit human blur decisions,
output inspection, correction, and hash-verified publication. Future data must
repeat those controls; the historical benchmark alone is not a release gate.
