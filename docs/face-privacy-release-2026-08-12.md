# Final Face-Privacy Pipeline and 2026-08-12 Release

This is the authoritative record of the face-screening, verification,
annotation, blurring, preservation, and Hugging Face publication completed in
August 2026. The detector benchmark and the earlier organization scan informed
the workflow, but their raw detections were not published as human-verified
face labels.

## Outcome

The release replaced 240 source videos on the `main` branch of five Hugging
Face dataset repositories with reviewed face-blurred derivatives:

| Repository | Videos inventoried | Deface 0.4 candidates | Qwen verified | Videos replaced on `main` |
|---|---:|---:|---:|---:|
| `Princeton_PRISM` | 1,008 | 413 | 13 | 10 |
| `Stanford_REAL` | 154 | 48 | 14 | 16 |
| `UPenn_PAL` | 424 | 178 | 26 | 1 |
| `UW_REAL` | 1,518 | 275 | 4 | 9 |
| `UofT_TISL` | 982 | 456 | 200 | 204 |
| **Total** | **4,086** | **1,370** | **257** | **240** |

One `UofT_TISL` source video was unreadable during screening. All 1,370
Deface-positive videos were successfully processed by Qwen. The 240 release
videos were selected by the manual privacy decision, not by treating the 257
Qwen-positive videos as an automatic publication list. The manual queue also
included strong Deface/Qwen disagreements and existing labeled examples; 22
released videos had been rejected by Qwen and three had no Qwen judgment.

## Finalized Workflow

1. **Repository triage.** An initial Deface 0.3 scan covered 9,990 videos in 15
   repositories after excluding `SOAR_origin`, `roboarena_upload`, and
   `Amazon-MobileManipulationTeam`. Manual review of up to five top candidates
   per repository found faces in the five repositories listed above. This was
   a prioritization heuristic, not proof that other repositories were
   face-free.
2. **Full-inventory five-repository screening.** All 4,086 videos in those five
   repositories were scanned with the CenterFace model bundled with
   ORB-HD/deface 1.5.0 at score threshold `0.4` and NMS threshold `0.4`. The
   presence scan stopped each video after its first positive frame and produced
   1,370 candidates.
3. **Semantic verification.** Every Deface-positive video was decoded again in
   full. Every detector-positive, full unannotated frame was sent to
   `Qwen/Qwen2.5-VL-7B-Instruct`, pinned to revision
   `cc594898137f460bfe9f0759e9844b3ce807cfb5`. Qwen verified 257 videos and
   rejected 1,113, with no failed or ambiguous videos. Qwen supplied a review
   filter only; it did not localize or blur faces.
4. **Manual privacy decisions.** A deduplicated 452-video queue combined the
   Qwen positives, high Deface/Qwen disagreements, the 101-video detector test
   set, and a 49-video Qwen review set. Every row received both a face-presence
   label and a downstream blur decision.
5. **Tracked-Deface rendering.** The 240 rows marked `blur` were rendered with
   native-resolution CenterFace detections, confirmed temporal tracking, and
   deface-style elliptical blur masks. Outputs were H.264 MP4s with copied
   audio and were validated against the source dimensions, frame count, frame
   rate, configuration, and output SHA-256.
6. **Conservative correction.** Ninety-seven suspect outputs were regenerated
   with a stricter fallback policy. All 97 passed machine validation; 14 were
   manually spot-checked and approved, while 83 were accepted without an
   additional individual correction-review label.
7. **Publication and preservation.** The final 240 derivatives and their
   release metadata were promoted to the datasets' `main` branches. Remote
   SHA-256 verification succeeded. Original media was retained in a private raw
   archive and through immutable pre-release tags.

## Manual Decisions

The annotation schema separates face presence from whether a face should be
blurred. The completed queue contained:

| Decision | Videos |
|---|---:|
| Face present | 290 |
| No face | 162 |
| Blur | 240 |
| Preserve | 212 |
| Preserve: already blurred | 24 |
| Preserve: task-relevant face depiction | 26 |

Thus, 50 face-containing videos were deliberately preserved. This prevents an
already blurred face from being processed unnecessarily and preserves human
face photographs or depictions that are part of the robot task. See
[`face-blur-annotations.md`](face-blur-annotations.md) for the schema and queue
composition.

## Rendering Configuration

The initial 240-video render used:

- ORB-HD/deface 1.5.0 CenterFace at native frame resolution;
- model SHA-256
  `09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc`;
- continuation threshold `0.2`, seed threshold `0.3`, and two observations to
  confirm a track;
- maximum gap of eight frames, two-frame pre-roll, and two-frame hold;
- deface-style elliptical masks at scale `1.3` with Gaussian blur;
- H.264 CRF 18, `medium` preset, and copied audio.

The 97-video correction retained this configuration and added a conservative
fallback track policy with two-frame pre-roll and 15-frame hold for strong
detections that did not form ordinary confirmed tracks. See
[`face-blur-pipeline.md`](face-blur-pipeline.md) and
[`deface-pipeline.md`](deface-pipeline.md) for implementation details.

## Hugging Face Data Layout

The five affected repositories expose the release at two levels:

- The root dataset card has the repository-level `face-blurred` tag. It means
  the repository contains some face-blurred videos, not that every video is
  blurred.
- Root-level `video_tags.csv` is the authoritative per-video index. Its columns
  are `video_path`, `tag`, `release`, `blurred_sha256`, and `config_hash`; the
  `tag` value is `face-blurred` for the 240 replaced paths.
- `face_blur_release/2026-08-12/` contains the per-repository release manifest,
  validation records, correction validation and review summary, and release
  README.

The final per-video tag counts are 10 for `Princeton_PRISM`, 16 for
`Stanford_REAL`, one for `UPenn_PAL`, nine for `UW_REAL`, and 204 for
`UofT_TISL`.

The dataset-card commits recorded after the release-scope clarification on
2026-08-22 were:

| Repository | Verified commit |
|---|---|
| `Princeton_PRISM` | [`362f159b`](https://huggingface.co/datasets/OopsieData-Submissions/Princeton_PRISM/commit/362f159b84d608fb0c7e339428e6d2e2e3abd7ae) |
| `Stanford_REAL` | [`1b509ae5`](https://huggingface.co/datasets/OopsieData-Submissions/Stanford_REAL/commit/1b509ae555c2b338a16bf02da4c98d339644379f) |
| `UPenn_PAL` | [`64624bf4`](https://huggingface.co/datasets/OopsieData-Submissions/UPenn_PAL/commit/64624bf4cee72e60cfe91e6e55721f6bfb0910e9) |
| `UW_REAL` | [`3f6d8aef`](https://huggingface.co/datasets/OopsieData-Submissions/UW_REAL/commit/3f6d8aef48ec4d2f69b2092d8b72013537b2a144) |
| `UofT_TISL` | [`ce36a76d`](https://huggingface.co/datasets/OopsieData-Submissions/UofT_TISL/commit/ce36a76dc81285b0652ec3c760868da40cd79aab) |

These commits are immutable audit points; later `main` commits may legitimately
move the repository head.

Raw originals are preserved in the private dataset
`OopsieData-Submissions/oopsie-face-raw-archive-2026-08-12` and by the
`pre-face-blur-2026-08-12` tag in each affected source repository. A default
checkout of `main` receives the blurred derivatives; access to the raw archive
must remain restricted.

## Recovering an Original Video

`pre-face-blur-2026-08-12` points to the last source revision recorded before
the 240 paths were replaced. To recover one original, use its repository and
the unchanged repository-relative `video_path` from `video_tags.csv`:

```bash
hf download OopsieData-Submissions/UofT_TISL \
  path/to/video.mp4 \
  --repo-type dataset \
  --revision pre-face-blur-2026-08-12 \
  --local-dir ./unblurred-recovery
```

The full repository can also be downloaded at that tag by omitting the video
path, although this transfers all files from that revision. A full source
commit hash from `release.json` can be supplied as `--revision` when an exact
commit is preferred over the named tag.

The tag does not create a separate permission boundary: anyone who can read an
affected source dataset can request its tagged revision and retrieve the
unblurred file. The private raw archive is the access-controlled backup, but it
does not make copies in source-repository history private. If originals must
be unavailable to ordinary source-repository readers, the tags and old source
history must be purged and recovery must be limited to the private archive.

## Scope and Limitations

- The release covers the August 2026 snapshots of the five listed
  repositories. It does not cover `SOAR_origin`, `roboarena_upload`,
  `Amazon-MobileManipulationTeam`, the other ten initially screened
  repositories, or repositories added later, such as `Brown_ST` and `NW_RZ`.
- No detector-negative or Qwen-negative decision proves that a video is
  face-free. The manual queue deliberately sampled important disagreements,
  but it did not exhaustively label every negative.
- The dataset-level `face-blurred` tag is a discoverability marker. Consumers
  needing exact paths must read `video_tags.csv`.
- The `pre-face-blur-2026-08-12` revision is intentionally recoverable by
  source-repository readers and contains the original unblurred media. It must
  not be described as access-controlled unless the source repository itself is
  access-controlled.
- New or changed videos require a new immutable inventory and another pass
  through screening, verification, manual decision, rendering, validation, and
  publication. The 2026-08-12 labels must not be silently applied to later
  repository revisions.

## Audit Records

The local release records used to verify these statements are:

```text
face_detection_results/oopsie-five-repos-deface-0.4/
face_detection_results/oopsie-five-repos-deface-0.4-qwen/
/datastor1/jiahuikchen/oopsie-face-annotations-20260811/annotations.csv
/datastor1/jiahuikchen/oopsie-face-blur-release-20260812/release.json
/datastor1/jiahuikchen/oopsie-face-blur-release-20260812/generated/validation.json
/datastor1/jiahuikchen/oopsie-face-blur-release-20260812/generated/publication.json
/datastor1/jiahuikchen/oopsie-face-blur-fallback-correction-20260812/review/
```

The large detector results, source videos, raw archive, and rendered outputs
are intentionally not stored in this Git repository.
