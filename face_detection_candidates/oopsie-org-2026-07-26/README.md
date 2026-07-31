# Oopsie Face-Presence Candidates — 2026-07-26

This directory publishes compact, collaborator-facing candidates from the
detector-only scan of the `OopsieData-Submissions` Hugging Face organization.
It intentionally excludes the 65,648 resumable per-video JSON checkpoints,
download caches, detector models, and all source or rendered video media.

## Important: Candidates, Not Verified Labels

A row means that `deface-centerface` produced at least one detection at score
threshold `0.3`. The scan stopped after the first positive frame, so these are
high-recall screening candidates rather than human-verified face labels.

Manual review found real faces in UofT_TISL, roboarena_upload, and at least one
Princeton_PRISM example, but also many false positives in other repositories.
Do not interpret repository totals as counts of confirmed faces.

## Files

- `face_candidate_videos.csv`: 24,310 videos with at least one detector trigger.
- `face_candidate_episodes.csv`: 18,317 filename-grouped episodes with at least
  one positive camera video.
- `config.json`: detector, model checksum, threshold, and stopping policy.
- `scan-report.json`: complete scan totals, candidate semantics, and failures.

Both CSVs retain the Hugging Face repository id and immutable repository
revision. `source_path` identifies the original video. Episode grouping is
marked `filename_heuristic` because the source repositories do not share a
uniform Hub-level episode-to-video index.

## Scan Totals

```text
Repositories with videos: 18
Videos inventoried:        65,648
Videos scanned:            65,647
Video candidates:          24,310
Episodes grouped:          48,336
Episode candidates:        18,317
Unreadable videos:         1
```

## Flagged Videos by Repository

The flagged videos are concentrated in the two bulk repositories:
`roboarena_upload` contributed 12,018 candidates and `SOAR_origin` contributed
7,424. Together they account for 19,442 of 24,310 flagged videos (80.0%). They
also account for 55,295 of 65,648 inventoried videos (84.2%), so this
concentration largely reflects their size and should not be read as a measure
of confirmed face prevalence.

The flag rate below is the percentage of successfully scanned videos that
triggered the detector. The table is ordered by number of flagged videos.

| Repository | Inventoried | Scanned | Flagged | Flag rate | Share of flagged |
|---|---:|---:|---:|---:|---:|
| `roboarena_upload` | 23,625 | 23,625 | 12,018 | 50.9% | 49.4% |
| `SOAR_origin` | 31,670 | 31,670 | 7,424 | 23.4% | 30.5% |
| `UofT_TISL` | 982 | 981 | 785 | 80.0% | 3.2% |
| `UW_REAL` | 1,518 | 1,518 | 621 | 40.9% | 2.6% |
| `Princeton_PRISM` | 1,008 | 1,008 | 610 | 60.5% | 2.5% |
| `Enatic_OpenArm` | 1,275 | 1,275 | 577 | 45.3% | 2.4% |
| `Princeton_IRoM` | 1,410 | 1,410 | 558 | 39.6% | 2.3% |
| `Berkeley_RAIL` | 1,499 | 1,499 | 447 | 29.8% | 1.8% |
| `UPenn_PAL` | 424 | 424 | 287 | 67.7% | 1.2% |
| `Amazon-MobileManipulationTeam` | 363 | 363 | 261 | 71.9% | 1.1% |
| `AI2_MOLMOACT2` | 225 | 225 | 158 | 70.2% | 0.6% |
| `CMU_Intent` | 585 | 585 | 149 | 25.5% | 0.6% |
| `UofT_RVL` | 126 | 126 | 121 | 96.0% | 0.5% |
| `UdeM_REAL` | 310 | 310 | 97 | 31.3% | 0.4% |
| `Stanford_REAL` | 154 | 154 | 82 | 53.2% | 0.3% |
| `Armnet` | 180 | 180 | 81 | 45.0% | 0.3% |
| `Princeton_VLL` | 54 | 54 | 33 | 61.1% | 0.1% |
| `Stanford_Iris` | 240 | 240 | 1 | 0.4% | <0.1% |
| **Total** | **65,648** | **65,647** | **24,310** | **37.0%** | **100.0%** |

The unreadable source is:

```text
OopsieData-Submissions/UofT_TISL
cups_converted/20260508_220901_episode_0_right_wrist.mp4
```

The MP4 lacks a `moov` atom and could not be decoded after two downloads. Its
episode remains an episode candidate because the left-wrist and top camera
videos both triggered the detector.
