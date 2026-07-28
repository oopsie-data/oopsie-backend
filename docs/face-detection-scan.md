# Hugging Face Face-Presence Scan

`scan_hf_faces` inventories the videos in one Hugging Face dataset or every
dataset visible under an organization, runs face detection, and writes
video- and episode-level CSV files. It does not blur, re-encode, or upload any
video.

## Does It Download the Videos?

The detector must decode the video frames, so every source video must be read
by the machine running the scan. If the data only exists on Hugging Face, that
means transferring its bytes.

The default `temporary` mode downloads one video at a time and deletes it as
soon as its result is saved. Disk use is therefore bounded to approximately
one source video plus the detector model and output metadata. This is not a
partial-video stream: each video is downloaded in full before OpenCV decodes
it.

Two alternatives are available:

- `--download-mode cache` retains the Hugging Face cache and avoids transferring
  the same immutable file on future runs.
- `--local-root` reads an existing organization sync and performs no new video
  download.

## Install

CPU:

```bash
uv sync --extra face-blur
```

GPU:

```bash
uv sync --extra face-blur-gpu
```

Set a Hugging Face token that can read every private dataset to be scanned:

```bash
export HF_TOKEN=hf_...
```

## Scan the Organization

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --org OopsieData-Submissions \
  --output-dir face_detection_results/oopsie-org \
  --device cuda:0
```

The default detector is `deface-centerface` at score threshold `0.3`. It is a
high-recall screening configuration: it detected all 30 positive videos in the
current labeled test set, but it also triggered on 30 of 58 labeled negative
videos. Treat `detected` as a candidate for review, not as a human-verified
label.

Scan one dataset instead:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --repo OopsieData-Submissions/UofT_TISL \
  --output-dir face_detection_results/uoft-tisl \
  --device cuda:0
```

For a quick end-to-end smoke test, limit both the inventory and decoded frames:

```bash
uv run --extra face-blur \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --repo OopsieData-Submissions/UofT_TISL \
  --output-dir face_detection_results/smoke \
  --device cpu \
  --limit 2 \
  --max-frames 50
```

`--max-frames` is for testing. A negative result from a truncated scan does not
establish that the full video has no face.

## Reuse an Existing Sync

The local root must use the same layout as `sync_hf_org`:
`<root>/<namespace>/<repo-name>/`.

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --org OopsieData-Submissions \
  --local-root /scratch/09666/abahety/oopsiedata \
  --output-dir face_detection_results/oopsie-org \
  --device cuda:0
```

To retain downloaded files in a dedicated cache instead:

```bash
uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --org OopsieData-Submissions \
  --download-mode cache \
  --cache-dir /scratch/$USER/oopsie-hf-cache \
  --output-dir face_detection_results/oopsie-org \
  --device cuda:0
```

## Outputs

```text
face_detection_results/oopsie-org/
  config.json
  inventory.csv
  videos.csv
  episodes.csv
  scan-report.json
  results/<video_id>.json
```

- `videos.csv` is the authoritative per-file inventory. Filter `has_face=True`
  for every video on which the detector triggered.
- `episodes.csv` combines camera videos that appear to belong to the same
  episode. Filter `face_status=detected` for the collaborator-facing episode
  list. `incomplete` means at least one video failed and none of the successful
  videos triggered.
- `results/` contains atomic per-video records. Finished results resume safely
  when the repository commit and detector configuration still match.
- `scan-report.json` records counts, detector/model configuration, source
  revision, and materialization mode.

By default the scan stops decoding a video as soon as one positive frame is
found. This makes the presence scan faster, but its positive-frame counts are
not exhaustive. Pass `--no-stop-on-positive` to scan every frame.

For a large organization, independent GPU jobs can divide the immutable
inventory into stable shards. First let one process write `inventory.csv`, then
pass that file to every worker so the workers do not repeat the Hub tree scan:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra face-blur-gpu \
  python -m oopsie_backend.scripts.scan_hf_faces \
  --inventory face_detection_results/oopsie-org/inventory.csv \
  --num-shards 3 --shard-index 0 \
  --output-dir face_detection_results/shard-0 --device cuda:0
```

Run indices `0`, `1`, and `2` with the same `--num-shards 3`, using a separate
output directory for each. Merge only after every shard completes:

```bash
uv run python -m oopsie_backend.scripts.merge_hf_face_scans \
  --input-dirs face_detection_results/shard-* \
  --output-dir face_detection_results/oopsie-org \
  --expected-videos 65648
```

The expected count makes the merge fail rather than silently publishing a
partial label file.

## Episode Mapping Limitation

The repositories do not expose one uniform Hub-level episode-to-video index.
The scanner therefore derives `episode_key` from the repo-relative filename by
removing known Oopsie camera suffixes. Every row is explicitly marked
`episode_mapping=filename_heuristic`, and the unmodified `source_repo` and
`source_path` remain in the output for auditing.

This is reliable for the existing one-file-per-episode-camera layouts. For a
dataset that stores multiple episodes in a single video, or uses an unfamiliar
camera naming convention, use `videos.csv` as the source of truth and resolve
its paths against that dataset's HDF5/Parquet metadata before treating
`episodes.csv` as exact.
