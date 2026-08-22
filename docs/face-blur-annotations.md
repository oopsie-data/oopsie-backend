# Manual Face and Blur-Policy Annotation

The finalized face-privacy workflow used a manual manifest that separates two
questions that must not be collapsed:

1. Does the video contain a human face or face depiction?
2. Should the downstream privacy pipeline blur it?

Qwen and Deface results were candidate evidence only. `blur_action` was the
release safety boundary: only rows explicitly marked `blur` were rendered and
published. The complete release context is in
[`face-privacy-release-2026-08-12.md`](face-privacy-release-2026-08-12.md).

## Annotation Schema

The final manifest stores:

```text
face_present = yes | no | uncertain
blur_action = blur | preserve | review
exemption_reason = already_blurred | task_relevant_depiction | other | <empty>
```

Valid completed decisions are:

| Meaning | `face_present` | `blur_action` | `exemption_reason` |
|---|---|---|---|
| Face should be blurred | `yes` | `blur` | empty |
| No face | `no` | `preserve` | empty |
| Face is already blurred | `yes` | `preserve` | `already_blurred` |
| Face/photo is task-relevant | `yes` | `preserve` | `task_relevant_depiction` |
| Other intentional preservation | `yes` | `preserve` | `other` |
| Cannot decide | `uncertain` | `review` | empty |

The annotation UI wrote the complete current state atomically to
`annotations.csv` and appended every manual edit, including before and after
values, to `annotations.history.jsonl`.

## Finalized Queue

The frozen workspace is
`/datastor1/jiahuikchen/oopsie-face-annotations-20260811`. Its 452 unique videos
combined:

- all 257 videos accepted by Deface 0.4 followed by Qwen2.5-VL;
- the 20 strongest Deface/Qwen disagreements per repository, ranked by the
  number of Deface-positive frames and then detection count;
- all 101 videos from the existing hand-labeled detector test set;
- all 49 videos from the August 2026 Qwen review set.

These sources overlap, so their raw sizes do not sum to 452. Existing test-set
labels and the 49-video review were imported before the remaining rows were
manually completed. Imported no-face labels could be completed as `preserve`.
Existing already-blurred examples and known task-relevant face depictions were
also preserved. A positive imported face label without an explicit blur policy
was shown as **Needs action** until the reviewer chose an action.

All 452 rows are complete:

| Field/value | Videos |
|---|---:|
| `face_present=yes` | 290 |
| `face_present=no` | 162 |
| `blur_action=blur` | 240 |
| `blur_action=preserve` | 212 |
| `exemption_reason=already_blurred` | 24 |
| `exemption_reason=task_relevant_depiction` | 26 |
| `uncertain` or `review` remaining | 0 |

The 212 preserved rows comprise 162 no-face videos and 50 face-containing
videos with explicit exemptions. Of the 240 blur rows, 215 were Qwen-verified,
22 were Qwen-rejected Deface disagreements, and three had no Qwen result. This
is why the 257 Qwen positives must not be used as a substitute for the final
manifest.

## Published Blur Decisions

The 240 `blur` decisions were distributed as follows:

| Repository | Videos blurred |
|---|---:|
| `Princeton_PRISM` | 10 |
| `Stanford_REAL` | 16 |
| `UPenn_PAL` | 1 |
| `UW_REAL` | 9 |
| `UofT_TISL` | 204 |

The release preparation copied the frozen annotation state into
`/datastor1/jiahuikchen/oopsie-face-blur-release-20260812/annotations.csv` and
recorded its source-manifest SHA-256 in `release.json`. The downstream renderer
selected only `blur_action=blur`; `preserve`, `review`, or incomplete rows were
not eligible.

Hugging Face consumers should use each repository's root `video_tags.csv` to
identify the published derivatives. The private annotation manifest contains
review evidence and local paths and is not itself the public per-video index.

## Policy for Future Runs

- Preserve the two-question schema. A face-presence label is not an instruction
  to blur task-relevant depictions or media that is already anonymized.
- Refuse release while any row is incomplete or has `blur_action=review`.
- Import prior labels only when source repository, path, and immutable revision
  match. A path-level label must not silently carry over to changed media.
- Keep the frozen manifest and history log with restricted release records;
  publish only the minimal per-video `face-blurred` index needed by dataset
  consumers.
