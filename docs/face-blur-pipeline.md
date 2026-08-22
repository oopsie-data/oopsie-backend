# Production Face-Blurring Pipeline

The finalized August 2026 release used the tracked-Deface pipeline, not the
earlier SCRFD implementation. It ran only on the 240 videos whose completed
manual annotation had `blur_action=blur`, wrote derived H.264 MP4s without
modifying the local raw sources, validated every output, and then replaced the
corresponding paths on Hugging Face `main`.

The authoritative end-to-end release record is
[`face-privacy-release-2026-08-12.md`](face-privacy-release-2026-08-12.md).
SCRFD remains available for detector research and historical comparison, as
documented in [`face-blur-benchmark.md`](face-blur-benchmark.md); it did not
render the 240 published derivatives.

## Release Input Boundary

Deface and Qwen outputs were not accepted directly as a blur list. The
completed 452-row manual manifest separated face presence from downstream
action. Only these rows were eligible:

```text
face_present=yes
blur_action=blur
exemption_reason=<empty>
```

This selected 240 videos. It excluded 162 reviewed no-face videos, 24 videos
that were already blurred, and 26 videos containing task-relevant face
depictions. See [`face-blur-annotations.md`](face-blur-annotations.md).

## Initial 240-Video Render

The release used ORB-HD/deface 1.5.0's CenterFace model and deface-style masks,
combined with temporal track confirmation:

| Setting | Released value |
|---|---|
| Detector | `deface-centerface` |
| Model SHA-256 | `09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc` |
| Detector resolution | Native source-frame resolution |
| Continuation threshold | `0.2` |
| Strong seed threshold | `0.3` |
| Observations required to confirm a track | `2` |
| Maximum association/interpolation gap | `8` frames |
| Confirmed-track pre-roll / hold | `2` / `2` frames |
| Smoothing window | `1` frame |
| Hold expansion | `0.02` per frame |
| Mask | Ellipse, scale `1.3`, Gaussian blur |
| Encoder | H.264, CRF 18, `medium` preset |
| Audio | Copied |

The planner works in two passes. CenterFace first runs over every frame; the
pipeline then confirms spatially consistent tracks, includes lower-scoring
continuations, interpolates short internal gaps, applies pre-roll and hold, and
renders the completed plan. It performs no face recognition and stores no face
embeddings.

A strong but unconfirmed detection was rendered on its observed frame rather
than discarded. The initial policy did not extend such fallback observations
as far as confirmed tracks, which motivated the correction pass below.

## Fallback Correction

An automated sidecar audit identified 97 outputs with fallback detections where
a conservative policy could cover frames that the initial confirmed-track
behavior might leave exposed. Those videos were regenerated with algorithm
version 3. The ordinary confirmed-track configuration stayed unchanged, while
fallback tracks received:

- two frames of pre-roll;
- up to 15 frames of post-detection hold.

All 97 corrected MP4s passed full machine validation. Fourteen were manually
spot-checked and approved; the remaining 83 had no additional individual
correction-review label and were accepted after validation. The corrected set
comprised eight `Princeton_PRISM`, four `Stanford_REAL`, one `UPenn_PAL`, nine
`UW_REAL`, and 75 `UofT_TISL` videos.

The correction pass replaced the corresponding initial derivatives before the
final 240-video set was promoted to `main`.

## Validation Contract

Each release output was required to have a successful audit sidecar and to
match the expected:

- source identity and source SHA-256;
- detector model and complete configuration hash;
- frame dimensions, decoded frame count, and frame rate;
- output encoding and successful full decode;
- output SHA-256.

The initial validation reported 240 of 240 valid videos and no errors. The
correction validation reported 97 of 97 valid videos and no errors. Publication
then compared the uploaded bytes with the local expected SHA-256 rather than
assuming a successful commit meant the media was correct.

## Hugging Face Publication

The final derivatives replaced the same repository-relative source paths on
each dataset's `main` branch:

| Repository | Replaced videos |
|---|---:|
| `Princeton_PRISM` | 10 |
| `Stanford_REAL` | 16 |
| `UPenn_PAL` | 1 |
| `UW_REAL` | 9 |
| `UofT_TISL` | 204 |
| **Total** | **240** |

Each affected dataset contains:

- root `video_tags.csv` with one `face-blurred` row per replaced path;
- a root dataset-card `face-blurred` tag for repository-level filtering;
- `face_blur_release/2026-08-12/` with manifest and validation metadata.

The dataset-card tag means “contains face-blurred videos.” It is not a claim
that all videos in the repository are blurred. Per-file consumers must use
`video_tags.csv`.

## Preservation and Recovery

Raw originals were preserved before replacement in two ways:

1. the private dataset
   `OopsieData-Submissions/oopsie-face-raw-archive-2026-08-12`, containing all
   240 original MP4s and a manifest;
2. the immutable `pre-face-blur-2026-08-12` tag in each affected source
   repository.

The default `main` revision now serves the face-blurred derivatives. Recovery
or research access to original media should use the access-controlled archive
or the immutable tag, never an unversioned local copy.

Recover one original by using the same repository-relative path at the
pre-release tag:

```bash
hf download OopsieData-Submissions/UofT_TISL \
  path/to/video.mp4 \
  --repo-type dataset \
  --revision pre-face-blur-2026-08-12
```

The `pre-face-blur-2026-08-12` tag contains unblurred media and inherits the
source repository's access policy. It is not made private by the existence of
the separate raw archive. See
[`Recovering an Original Video`](face-privacy-release-2026-08-12.md#recovering-an-original-video)
for the security implications and exact source-revision alternative.

## Running the Core Renderer

Install one ONNX Runtime variant and the FFmpeg-backed encoder:

```bash
# CPU
uv sync --extra face-blur

# GPU
uv sync --extra face-blur-gpu
```

The tracked renderer is exposed through:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.run_tracked_deface \
  --input /path/to/reviewed-video.mp4 \
  --output-dir /path/to/derived-videos \
  --device cuda:0
```

The CLI defaults reproduce the initial release's core detector, tracking,
mask, encoder, and audio settings. Reproducing a dataset release additionally
requires a frozen action-filtered manifest, immutable source revisions, the
fallback correction policy, validation, raw preservation, and remote hash
verification. A successful standalone render is not authorization to replace
Hub data.

## Review Requirements for Future Releases

Before publication, inspect faces around entry, exit, occlusion, profile views,
fast motion, reflections, and screen or printed depictions. Record whether a
depiction is privacy-sensitive or task-relevant rather than letting the
detector decide policy. Any suspected leak should enter a correction manifest;
new or changed source videos must repeat the full screening and review
workflow.
