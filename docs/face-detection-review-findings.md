# Face-Detection Screening and Manual-Review Findings

Date: 2026-08-02

This note records the preliminary face-presence screening results and a manual
review of the highest-scoring Deface candidates. These findings were used to
prioritize five repositories for the finalized pipeline; they are not
exhaustive human-verified labels.

**Status:** historical triage record. It is superseded as a release description
by [`face-privacy-release-2026-08-12.md`](face-privacy-release-2026-08-12.md).
The final workflow rescanned the five selected repositories at threshold `0.4`,
ran pinned Qwen verification, completed a 452-video manual queue, and published
240 reviewed face-blurred derivatives.

## Scope

The July 2026 Hugging Face inventory contained 65,648 videos across 18
repositories. This review excludes the two bulk repositories, `SOAR_origin`
and `roboarena_upload`, and `Amazon-MobileManipulationTeam`. The resulting
scope contains 9,990 videos across 15 repositories. Of these, 9,989 decoded
successfully; one malformed `UofT_TISL` MP4 could not be read.

## Deface Screening Configuration

The primary screening results came from the CenterFace model bundled with
ORB-HD/deface v1.5.0 (`deface-centerface`):

- score threshold: `0.3`
- NMS threshold: `0.4`
- minimum positive frames: `1`
- minimum consecutive positive frames: `1`
- stop after the first positive frame: enabled

A positive row therefore means that CenterFace triggered on at least one
decoded frame. It does not mean that a person manually verified a face.

Deface flagged 4,607 of the 9,990 in-scope videos as face candidates.

## Manual Review of Top Candidates

Up to five deface-positive videos were selected from each of the 15
repositories. Candidates were ordered by their recorded detector score, with
at most one camera video selected from each inferred episode. `Stanford_Iris`
had only one deface-positive video, so this produced 71 review videos rather
than 75.

Because the source scan stopped at the first positive frame, the ranking score
is the maximum detection score on that first triggering frame. It is not an
exhaustive maximum over every frame in the video.

Faces were observed in the reviewed samples from five repositories:

- `Princeton_PRISM`
- `Stanford_REAL`
- `UofT_TISL`
- `UPenn_PAL`
- `UW_REAL`

No faces were observed in the reviewed top candidates from the other ten
repositories:

- `AI2_MOLMOACT2`
- `Armnet`
- `Berkeley_RAIL`
- `CMU_Intent`
- `Enatic_OpenArm`
- `Princeton_IRoM`
- `Princeton_VLL`
- `Stanford_Iris`
- `UdeM_REAL`
- `UofT_RVL`

This result suggests that the deface triggers in the latter repositories may
be dominated by systematic false positives. It does **not** establish that
those repositories are face-free: the review covered at most five
detector-positive videos per repository, did not randomly sample all positive
videos, and did not examine detector-negative videos for missed faces.

## Deface Candidates in Repositories with Observed Faces

Restricting this Deface 0.3 candidate list to the five repositories where faces
were observed gives 2,385 candidate videos:

| Repository | Inventoried | Successfully scanned | Deface candidates |
|---|---:|---:|---:|
| `Princeton_PRISM` | 1,008 | 1,008 | 610 |
| `Stanford_REAL` | 154 | 154 | 82 |
| `UofT_TISL` | 982 | 981 | 785 |
| `UPenn_PAL` | 424 | 424 | 287 |
| `UW_REAL` | 1,518 | 1,518 | 621 |
| **Total** | **4,086** | **4,085** | **2,385** |

These 2,385 videos were still detector candidates rather than individually
verified positives. Restricting to the five repositories is a practical
review heuristic based on the observed sample, not a validated classifier.

## Subsequent Finalized Scan and Release

The five repositories were rescanned at Deface threshold `0.4`. This later
immutable scan, rather than the 2,385-candidate table above, was the input to
the finalized Qwen stage:

| Repository | Videos | Deface 0.4 candidates | Qwen verified | Published blur actions |
|---|---:|---:|---:|---:|
| `Princeton_PRISM` | 1,008 | 413 | 13 | 10 |
| `Stanford_REAL` | 154 | 48 | 14 | 16 |
| `UofT_TISL` | 982 | 456 | 200 | 204 |
| `UPenn_PAL` | 424 | 178 | 26 | 1 |
| `UW_REAL` | 1,518 | 275 | 4 | 9 |
| **Total** | **4,086** | **1,370** | **257** | **240** |

The final blur count can be greater or smaller than the Qwen-positive count in
a repository because human review also considered strong Deface/Qwen
disagreements and explicit exemptions. The manual action, not Qwen, determined
publication.

## Comparison with InsightFace SCRFD-10G

SCRFD-10G from InsightFace was also run at threshold `0.3` over the same
9,990-video scope. The two detectors produced:

| Detection outcome | Videos |
|---|---:|
| Both detectors positive | 3,896 |
| Deface positive overall | 4,607 |
| SCRFD-10G positive overall | 6,705 |
| Deface only | 711 |
| SCRFD-10G only | 2,809 |
| Neither detector positive | 2,573 |
| Unreadable | 1 |

Thus, 84.6% of deface candidates were also detected by SCRFD-10G, while only
58.1% of SCRFD-10G candidates were also detected by deface. This is consistent
with deface being more selective on these videos, but it is not by itself a
precision measurement because most candidates do not have human labels.

## Existing Verified Test Set

The separate hand-labeled face-blur test set contains 30 verified-positive
videos:

- 27 from `UofT_TISL`
- 2 from `UPenn_PAL`
- 1 from `UW_REAL`

It contains no verified-positive examples from most repositories. The formal
101-video detector-test manifest itself was not rewritten with the top-candidate
observations, but all 101 test videos and the later 49-video Qwen review set were
imported into the separate 452-video final privacy annotation queue.

## Recommended Interpretation

Use the following wording when sharing these results:

> Manual review found faces among the five highest-scoring, distinct-episode
> deface candidates from Princeton_PRISM, Stanford_REAL, UofT_TISL, UPenn_PAL,
> and UW_REAL. No faces were observed in the corresponding reviewed samples
> from the other ten repositories. The unreviewed videos have not been proven
> face-free.

For privacy-sensitive decisions, review a larger random sample of both
Deface-positive and Deface-negative videos before declaring any repository
face-free. The finalized 240-video release addressed the selected candidates;
it did not establish that excluded, model-negative, newly added, or subsequently
changed videos contain no faces.

## Source Artifacts

- Deface organization scan: `face_detection_results/oopsie-org/videos.csv`
- Published deface candidates:
  `face_detection_candidates/oopsie-org-2026-07-26/face_candidate_videos.csv`
- SCRFD-10G comparison scan:
  `face_detection_results/oopsie-org-scrfd-10g/videos.csv`
- Hand-labeled test set: `face_blur_testset/manifest.csv`
- Final Deface 0.4 scan:
  `face_detection_results/oopsie-five-repos-deface-0.4/videos.csv`
- Final Qwen verification:
  `face_detection_results/oopsie-five-repos-deface-0.4-qwen/videos.csv`
- Final publication record:
  `/datastor1/jiahuikchen/oopsie-face-blur-release-20260812/generated/publication.json`
