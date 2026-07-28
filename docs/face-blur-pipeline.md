# Face-Blurring Pipeline

The face-blurring stage runs SCRFD on every decoded video frame, associates
detections across time, fills short detector gaps, and writes a new H.264 MP4.
It never modifies the source video in place.

The default operating point comes from the detector benchmark:

- SCRFD-10G at score threshold `0.3`
- `640x640` detector input
- up to 8 missing frames associated into the same track
- linear box interpolation between detections
- 2-frame pre-roll before a track's first detection
- 8-frame motion-projected hold after its last detection
- 4% additional hold-box expansion per missing frame
- 25% horizontal, 35% top, and 25% bottom face-box padding
- two-pass strong Gaussian blur

SCRFD-34G is also registered for experiments, but it is not the default. On
the current benchmark set it produced more false-positive robot/workspace
regions than SCRFD-10G at the highest tested threshold that retained 30/30
video-level positive recall.

The pre-roll, interpolation, and post-detection hold are possible because each
video is processed in two passes. The first pass completes the track plan; the
second pass renders it. A one-frame detector miss between two observations is
therefore filled from both neighboring boxes rather than copied blindly from
the previous frame.

## Install

Use one of the existing face-blur extras. Both include the FFmpeg encoder used
to produce browser-compatible H.264 output.

```bash
# CPU inference
uv sync --extra face-blur

# GPU inference
uv sync --extra face-blur-gpu
```

Do not install both extras in one environment because both ONNX Runtime
distributions expose the same Python module.

## Process the 30 Positive Review Videos

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.blur_faces \
  --manifest face_blur_testset/manifest.csv \
  --labels yes \
  --output-dir face_blur_outputs/positive-review \
  --device cuda:0
```

The generated videos are under
`face_blur_outputs/positive-review/videos/`. The directory also contains a
`face-blur-run.json` batch report. Every video has an adjacent
`<video>.face-blur.json` audit sidecar with:

- source and output SHA-256 hashes;
- detector/model/configuration metadata;
- frame and timing counts;
- track observations and scores;
- counts of directly detected, interpolated, pre-roll, and held regions.

The command resumes videos whose source stat and complete processing
configuration still match. An existing output with missing or stale metadata
is not overwritten unless `--overwrite` is given. Outputs are encoded and
validated in a temporary file, then published with an atomic rename.

## Process Other Data

One video:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.blur_faces \
  --input /path/to/camera.mp4 \
  --output-dir /path/to/derived-videos \
  --device cuda:0
```

A directory tree, preserving relative paths:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.blur_faces \
  --input /path/to/dataset/videos \
  --output-dir /path/to/blurred-dataset/videos \
  --device cuda:0
```

The library entry point is
`oopsie_backend.face_blur.pipeline.blur_video`, so conversion jobs can call the
same implementation directly after writing each camera video. Keep originals
in a restricted raw-data location and publish only the derived output.

Native and confirmed-tracking ORB-HD/deface comparisons are kept as separate
pipelines. Both use CenterFace and ellipse masks; the tracked variant confirms
tracks with strong detections before accepting weaker temporal continuations.
See
[`deface-pipeline.md`](deface-pipeline.md).

## Temporal Behavior

Detections are matched one-to-one using predicted motion, box overlap, center
distance, and size consistency. The tracker does not use face recognition or
store embeddings.

For each track:

1. Nearby detector boxes are temporally averaged, then unioned with the raw
   boxes. Smoothing can enlarge a region but cannot remove detector-supported
   coverage.
2. Missing boxes between two matched detections are linearly interpolated.
3. The box is projected for 8 frames after the last detection using recent
   center/size velocity with damping.
4. Held boxes expand on every missed frame to account for motion uncertainty.
5. All planned boxes receive additional padding before rendering.

Every accepted detection creates a blur immediately; there is no minimum-hit
filter that could expose a real face appearing for only one frame. A false
positive can consequently persist for the configured hold period. This is an
intentional privacy/utility tradeoff.

Relevant controls include:

| Option | Default | Meaning |
|---|---:|---|
| `--score-threshold` | `0.3` | SCRFD acceptance threshold |
| `--max-gap-frames` | `8` | Maximum missing frames still associated/interpolated |
| `--hold-frames` | `8` | Frames blurred after the last detection |
| `--pre-roll-frames` | `2` | Frames blurred before the first detection |
| `--smoothing-window` | `2` | Temporal smoothing radius in frames |
| `--hold-expansion-per-frame` | `0.04` | Extra scale per held frame |
| `--blur-method` | `gaussian` | `gaussian` or `pixelate` |
| `--audio` | `copy` | Preserve audio, re-encode to AAC, or `drop` |

Use `python -m oopsie_backend.scripts.blur_faces --help` for the complete set.

## Manual Validation

Review every positive output from beginning to end, at normal speed and with
frame stepping around each face's entry, exit, occlusion, and fast motion.
Check specifically for:

- any unblurred face frame, including profiles and partial faces;
- a box that lags behind a moving face during interpolation or hold;
- insufficient forehead, chin, or side padding;
- blur strength that leaves identity recognizable;
- task-relevant printed/screen faces that need a documented policy decision;
- output truncation, frame-size changes, corruption, or audio problems.

The current video-level test labels cannot prove frame-level coverage. The 30
video review is therefore a required validation step, not an optional visual
spot check. Record any leaking sample ID and frame/timestamp before tuning so a
regression fixture can be added.
