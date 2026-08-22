# Face-Blur Test Set

This document describes the hand-labeled video set assembled on the
`face-blur` branch to seed development and evaluation of a face detection and
blurring pipeline.

This 101-video set is the finalized **detector-development set**, not the final
publication manifest. It was imported into the later 452-video manual privacy
queue, but the 240 Hugging Face replacements were selected by that queue's
separate `blur_action` field. See
[`face-blur-annotations.md`](face-blur-annotations.md) and the
[`2026-08-12 release record`](face-privacy-release-2026-08-12.md).

## Final Detector Test Set

Current location:

```text
face_blur_testset/
  manifest.csv
  videos/
```

The final pruned set contains `101` videos. The manifest and video directory
are consistent: every manifest row has one local video file and there are no
orphan files in `videos/`.

The test-set directory now lives inside this repository. `manifest.csv` and
`README.md` are intended to be committed; `videos/` is intentionally ignored by
git because it contains the local video media.

Label distribution from `manifest.csv`:

```text
yes:       30
no:        58
uncertain: 13
total:    101
```

Source breakdown:

```text
OopsieData-Submissions/UofT_TISL:                    27 yes, 43 no, 13 uncertain
OopsieData-Submissions/UW_REAL:                       1 yes,  6 no
OopsieData-Submissions/UPenn_PAL:                     2 yes,  1 no
OopsieData-Submissions/Berkeley_RAIL:                         3 no
OopsieData-Submissions/Amazon-MobileManipulationTeam:         5 no
```

The `13` `uncertain` rows are all UofT_TISL videos where faces were already
blurred before this test set was assembled. They all have the note
`Face already blurred`.

## Manifest Schema

`manifest.csv` has these columns:

```text
sample_id,source_kind,source_repo,source_path,local_video_path,face_present,labeler,notes
```

Important fields:

- `sample_id`: stable local id, for example `video_0077`.
- `source_kind`: currently `huggingface` for the final set.
- `source_repo`: Hugging Face dataset repo id.
- `source_path`: original path inside the source repo.
- `local_video_path`: path relative to the test-set root, under `videos/`.
- `face_present`: hand label, one of `yes`, `no`, or `uncertain`.
- `notes`: free-form edge-case notes from labeling.

The recommended evaluation split is:

- Use `yes` as positive examples for face detection.
- Use `no` as negative examples.
- Report `uncertain` separately because these are already-blurred faces, not
  clean negatives or clean positives.

## Label Semantics

The binary label asks whether the video contains face-like human visual content
that a face detector/blurring pipeline may need to handle.

Use:

- `yes`: at least one visible human face, printed/photo face, screen face, or
  other face-like human image appears in the video.
- `no`: no visible human face-like content appears.
- `uncertain`: a face appears to have already been blurred, or the labeler
  could not confidently decide.

For privacy-oriented blur evaluation, do not treat all `yes` rows as equivalent.
Some positives are real people in the scene, while others are printed or
screen-displayed faces that may be task-relevant objects.

Known edge cases:

- `video_0172` from `OopsieData-Submissions/UPenn_PAL` includes printed/photo
  faces of public figures. The paired H5 task instruction is
  `put the pineapple in the bowl closer to Sam Altman`, so face identity is
  required for task completion. Blurring this face would remove task-relevant
  information even though a face detector should detect it.
- The `13` UofT_TISL `uncertain` examples are already-blurred faces. They are
  useful for checking idempotence or for measuring how the pipeline behaves on
  pre-blurred data, but they should not be counted as ordinary clean positives.

The finalized release implemented that separation in its annotation manifest:

```text
face_present = yes | no | uncertain
blur_action = blur | preserve | review
exemption_reason = already_blurred | task_relevant_depiction | other | <empty>
```

The release used `face_present`, `blur_action`, and `exemption_reason` as the
authoritative fields. Twenty-four already-blurred videos and 26 task-relevant
face depictions were explicitly preserved; neither category was inferred from
the detector label alone.

## Scripts Added On This Branch

Two lightweight utilities were added for this work:

```text
oopsie_backend/scripts/sample_face_blur_videos.py
oopsie_backend/scripts/label_face_blur_videos.py
```

The sampler creates or appends to a manifest and copies sampled Hugging Face
videos into `videos/`. In append mode it excludes videos already present in the
manifest by matching `source_kind`, `source_repo`, and `source_path`.

The labeler serves a local web UI and edits the manifest in place. It supports
HTTP byte ranges so videos can be seeked in the browser.

## Sampling More Videos

Export a Hugging Face token with access to the private OopsieData datasets:

```bash
export HF_TOKEN=hf_...
```

The token value must be only the `hf_...` token string. Do not include labels
such as `Oopsie-Data Master Key Token:`.

Sample from a Hugging Face dataset:

```bash
uv run python -m oopsie_backend.scripts.sample_face_blur_videos \
  --repo OopsieData-Submissions/UofT_TISL \
  --output-dir face_blur_testset \
  --count 20 \
  --append
```

By default, sampling uses seed `20260714`. Passing a different `--seed` changes
which videos are selected. In append mode, previously sampled source paths from
the same repo are excluded before sampling.

Sample from an already-synced local dataset directory:

```bash
uv run python -m oopsie_backend.scripts.sample_face_blur_videos \
  --local-dir /corral/utexas/IRI26024/oopsiedata/OopsieData-Submissions/UofT_TISL \
  --output-dir face_blur_testset \
  --count 20 \
  --append
```

Local samples are symlinked by default. Pass `--copy` if the new set must be
portable.

## Labeling

Launch the local annotation tool:

```bash
uv run python -m oopsie_backend.scripts.label_face_blur_videos \
  --manifest face_blur_testset/manifest.csv \
  --host 0.0.0.0 \
  --port 9876
```

If using VS Code Remote SSH, forward the same port in VS Code and open the
forwarded localhost URL in the local browser. If a forwarded port appears stale
or shows "no running process", restart the labeler on a fresh port such as
`9876`; previous trouble was caused by stale/confused forwarding on `8765`, not
by the labeler itself.

Keyboard shortcuts:

```text
y: face present
n: no face
u: uncertain
right arrow: next video
left arrow: previous video
```

The browser UI can also edit `labeler` and `notes`.

## Cleanup Commands

After annotation, remove unlabelled rows and their videos before treating the
set as final:

```bash
uv run python - <<'PY'
import csv
import shutil
from datetime import datetime
from pathlib import Path

root = Path('face_blur_testset')
manifest = root / 'manifest.csv'
backup = root / f"manifest.before-prune-unlabeled-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
shutil.copy2(manifest, backup)

with manifest.open(newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

kept = []
removed = []
for row in rows:
    if (row.get('face_present') or '').strip():
        kept.append(row)
    else:
        removed.append(row)

for row in removed:
    rel = row.get('local_video_path') or ''
    if rel:
        (root / rel).unlink(missing_ok=True)

with manifest.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept)

print(f'backup={backup}')
print(f'removed={len(removed)}')
print(f'kept={len(kept)}')
PY
```

Check consistency:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

root = Path('face_blur_testset')
with (root / 'manifest.csv').open(newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))

manifest_files = {
    root / (row.get('local_video_path') or '')
    for row in rows
    if row.get('local_video_path')
}
disk_files = {
    path
    for path in (root / 'videos').glob('*')
    if path.is_file()
}

print('manifest_rows', len(rows))
print('disk_video_files', len(disk_files))
print('missing_from_disk', sum(1 for path in manifest_files if not path.exists()))
print('orphan_disk_files', sum(1 for path in disk_files if path not in manifest_files))
PY
```
