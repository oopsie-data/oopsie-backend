# Qwen2.5-VL Face Verification

`verify_hf_faces_qwen` is the semantic verification stage used after Deface in
the finalized August 2026 face-privacy workflow. It uses Qwen2.5-VL as an
additional filter after
`deface-centerface`; it does not replace the face detector, generate bounding
boxes, blur media, or upload anything.

The verifier's output is a review-candidate signal, not the final privacy
decision. The completed release used manual `face_present` and `blur_action`
labels before any media was replaced. See the authoritative
[`2026-08-12 release record`](face-privacy-release-2026-08-12.md).

The verifier uses
[`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
pinned by default to the exact model revision used in the evaluated run
(`cc594898137f460bfe9f0759e9844b3ce807cfb5`). Override `--qwen-revision`
only when intentionally evaluating a different model snapshot.

## Decision Policy

The input must be a completed deface Hugging Face scan. The verifier:

1. selects videos marked positive by that scan;
2. verifies the detector model, input size, NMS threshold, and score threshold,
   then reruns deface through every frame with those same settings;
3. sends every deface-positive **full, unannotated frame** to Qwen;
4. asks only whether a visible human face is present;
5. marks a video verified if Qwen answers `YES` for at least one frame.

The prompt counts real faces and human faces in photographs, posters,
reflections, and screens. It rejects robot parts, grippers, isolated hands and
arms, tools, shadows, and accidental face-like patterns. Qwen is not asked to
localize the face. The full frame is used rather than a detector crop so the
model retains scene context and is not forced to interpret the false-positive
region as a face. A previously blurred region with no discernible facial
features is answered `NO`; the later manual policy separately records
`already_blurred` as a face-present preservation exemption.

Every raw Qwen response is retained. An ambiguous response produces an
`incomplete` result rather than being silently converted to `NO`.
An unexpectedly short video decode or a detector-positive source video with no
positive frames on the exhaustive rerun is also `incomplete`, never rejected.

## Evaluated Production Run

The finalized run consumed the Deface 0.4 scan of 4,086 videos from
`Princeton_PRISM`, `Stanford_REAL`, `UPenn_PAL`, `UW_REAL`, and `UofT_TISL`.
Deface produced 1,370 candidates, all of which completed Qwen verification:

| Result | Videos |
|---|---:|
| Qwen verified | 257 |
| Qwen rejected | 1,113 |
| Incomplete or failed | 0 |
| Verified episodes by filename heuristic | 207 |

Qwen checked 52,694 Deface-positive frames and answered `YES` on 30,841. These
figures describe the verifier, not the published dataset. Manual review later
selected 240 videos for blurring: 215 Qwen-verified videos, 22 Qwen-rejected
Deface disagreements, and three videos without a Qwen judgment. Forty-two
Qwen-verified videos were preserved after manual review.

## Installation

Install the GPU face detector and Qwen runtime together:

```bash
uv sync --extra face-blur-gpu --extra face-verify-qwen
```

The Qwen worker is a separate Python process. `--worker-python` can point to a
different environment containing PyTorch, Transformers, Accelerate, and
Pillow, which allows the detector and VLM dependencies to remain isolated.

## Run

First produce a completed deface scan at the desired threshold. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  -m oopsie_backend.scripts.verify_hf_faces_qwen \
  --detector-results-dir \
    face_detection_results/oopsie-five-repos-deface-0.4 \
  --output-dir \
    face_detection_results/oopsie-five-repos-deface-0.4-qwen \
  --detector-threshold 0.4 \
  --qwen-cache-dir /path/to/huggingface/hub/cache \
  --batch-size 4
```

To use a separate worker environment:

```bash
  --worker-python /path/to/qwen/environment/bin/python
```

The source video is downloaded one at a time and deleted after processing.
Detector-positive frames are JPEG-encoded in memory and sent to the persistent
worker over a local pipe. They are not written as review images.

## Sharding

Independent GPU jobs can split the detector-positive video set deterministically:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  -m oopsie_backend.scripts.verify_hf_faces_qwen \
  --detector-results-dir face_detection_results/oopsie-five-repos-deface-0.4 \
  --output-dir face_detection_results/qwen-shard-0 \
  --detector-threshold 0.4 \
  --num-shards 3 --shard-index 0
```

After all shards finish, merge them with an exact expected count:

```bash
uv run python -m oopsie_backend.scripts.merge_qwen_face_verifications \
  --input-dirs face_detection_results/qwen-shard-* \
  --output-dir face_detection_results/oopsie-five-repos-deface-0.4-qwen \
  --expected-videos 1370
```

## Outputs

```text
output-dir/
  config.json
  scan-report.json
  videos.csv
  episodes.csv
  results/<video_id>.json.gz
```

- `videos.csv` contains the deface-positive frame count, Qwen-positive,
  Qwen-negative, and unparsed frame counts, and the final video judgment.
- `episodes.csv` aggregates candidate camera videos using the existing
  filename-derived episode key.
- `results/` contains the deface boxes and scores plus Qwen's raw response for
  every checked frame. It contains no frame pixels.

Results are atomic and resumable against the immutable Hugging Face revision,
detector configuration, resolved Qwen model revision, prompt, visual-token
limits, JPEG quality, and generation settings.

## Limitations

Qwen2.5-VL is not a validated privacy detector. A `NO` response can still be a
false negative, especially for small, occluded, profile, low-resolution, or
motion-blurred faces. The August release therefore included high Deface/Qwen
disagreements in the manual queue and used manual blur decisions as the safety
boundary. For future runs, ambiguous or failed verification must remain review
candidates, and a model `NO` must not be treated as proof that a video is
face-free.
