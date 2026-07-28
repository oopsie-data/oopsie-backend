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

The unreadable source is:

```text
OopsieData-Submissions/UofT_TISL
cups_converted/20260508_220901_episode_0_right_wrist.mp4
```

The MP4 lacks a `moov` atom and could not be decoded after two downloads. Its
episode remains an episode candidate because the left-wrist and top camera
videos both triggered the detector.
