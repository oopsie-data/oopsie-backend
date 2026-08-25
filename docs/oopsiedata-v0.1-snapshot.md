# OopsieData v0.1 Snapshot Release

This document records the construction and publication of the private
`OopsieData-Submissions/oopsiedata-v0.1` Hugging Face dataset. The repository
is a self-contained, lab-organized snapshot of 19 source repositories. It is
not a live mirror: later changes to the source repositories do not alter the
immutable `v0.1` tag.

## Final Release

| Property | Value |
|---|---|
| Destination | `OopsieData-Submissions/oopsiedata-v0.1` |
| Visibility | Private pending collaborator review |
| Immutable tag | `v0.1` |
| Validated commit | `1b5c9171fc28edf82e1e375fae3d14d66d18a45f` |
| Frozen input time | `2026-08-22T17:28:38-05:00` |
| Source repositories | 19 |
| Videos | 26,068 |
| Release bytes | 51,790,530,584 (48.23 GiB) |
| Face-blurred videos | 394 |
| Preserved face-containing exemptions | 50 |

Every one of the 26,068 remote MP4s was checked against its expected byte
count and SHA-256 before the tag was created. `main` and `v0.1` pointed to the
same validated commit at completion.

## Scope

The scan froze each repository at an exact revision:

| Repository | Videos | Frozen revision |
|---|---:|---|
| `Araya` | 14,400 | `d8ae0a9337ae9b8870e65227e6d69e46c84a6dd7` |
| `Armnet` | 180 | `b0502bb1384c092c19ff5826cf3c74b048714774` |
| `Berkeley_RAIL` | 1,499 | `113d1df17318eaaf3e368d47b49a3443a004c9a7` |
| `CMU_Intent` | 585 | `9de886bbfb0684a8651399013852d649800e88da` |
| `Enatic_OpenArm` | 1,275 | `b7a12f9d8d61b03768683452190a68956b950ebb` |
| `Harvard_EML` | 814 | `11c2c3fb2cb2268cd3208d784128ad0570ec9c6c` |
| `Princeton_IRoM` | 1,410 | `0c9db002067ca472e9b8cc93648a1e603c17bfa3` |
| `Princeton_PRISM` | 1,008 | `362f159b84d608fb0c7e339428e6d2e2e3abd7ae` |
| `Princeton_PRPL` | 1,074 | `03189526db5e14959fe88946b1f63f994f5cb739` |
| `Princeton_VLL` | 54 | `17a28f7b8f7e03351e376067b0b837ccb6a61f76` |
| `Stanford_Iris` | 240 | `074c5684d89b35436ae634cffcff461bf61cea82` |
| `Stanford_MSL` | 9 | `aee5d6cecade9878ebdccd79f42724a35548a85c` |
| `Stanford_REAL` | 154 | `1b509ae555c2b338a16bf02da4c98d339644379f` |
| `UPenn_PAL` | 424 | `64624bf4cee72e60cfe91e6e55721f6bfb0910e9` |
| `UT_RobIn` | 6 | `b239396ff31323185e49520aafe5cb98f1e7e13f` |
| `UW_REAL` | 1,518 | `3f6d8aef48ec4d2f69b2092d8b72013537b2a144` |
| `UdeM_REAL` | 310 | `60340c8370b994ea1a60835ba83c208a5c43727a` |
| `UofT_RVL` | 126 | `8bbe3ea2dcc40eab95ed74cc2cb794f516744d83` |
| `UofT_TISL` | 982 | `ce36a76dc81285b0652ec3c760868da40cd79aab` |

The release intentionally excludes the converted datasets
`AI2_MOLMOACT2`, `Amazon-MobileManipulationTeam`, `SOAR_origin`, and
`roboarena_upload`. It also excludes the non-source `Visualizer_Previews`,
`WebVisualizer`, and private raw-archive repositories.

The authoritative frozen input is:

```text
face_detection_results/oopsie-submissions-2026-08-22/scope/inventory.csv
```

`release.json` in the snapshot repeats the 19 repository revisions and counts,
so a separate `source_repositories.csv` is unnecessary.

## Privacy Decisions

The full inventory was screened with Deface CenterFace at score threshold
`0.4` and NMS threshold `0.4`. All 5,665 Deface-positive videos were then sent
through `Qwen/Qwen2.5-VL-7B-Instruct` at pinned revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`.

The final release decisions were:

| Privacy status | Videos |
|---|---:|
| Blurred | 394 |
| Deface not detected | 20,232 |
| Qwen rejected | 5,234 |
| Manually reviewed no face | 157 |
| Exempt: already blurred | 24 |
| Exempt: task-relevant depiction | 26 |
| Scan error | 1 |

The 394 blurred videos comprise:

- 240 manually reviewed derivatives already present in their frozen source
  repository revisions;
- 115 new Qwen-verified positives;
- 19 high-risk Qwen rejections selected by the strict disagreement policy;
- 20 additional Qwen rejections selected by the broader camera policy.

The 115, 19, and 20 groups were rendered locally with the finalized tracked
Deface algorithm-3 policy. They use native-resolution CenterFace detection,
two-frame fallback pre-roll, 15-frame fallback hold, elliptical masks at scale
1.3, H.264 CRF 18, and copied audio. See
[`face-blur-pipeline.md`](face-blur-pipeline.md).

The 50 exemptions remain byte-identical to their frozen source files. A
detected face is not sufficient to authorize blurring when it was already
obscured or was a task-relevant photograph or depiction.

## Unreadable UofT Video

`UofT_TISL/cups_converted/20260508_220901_episode_0_right_wrist.mp4`
could not be decoded because the 262,192-byte MP4 has no readable `moov` atom.
It therefore has `privacy_status=scan_error`; neither Deface nor Qwen could
evaluate its frames.

The file was retained for exact frozen-snapshot completeness at:

```text
labs/UofT_TISL/cups_converted/20260508_220901_episode_0_right_wrist.mp4
```

Its release SHA-256 is
`d0a38b25c1ca7a9cd2be64b9d75be2f5cafd82c60e5d04f6b497ffce9641a94d`.
Removing or replacing it is a data-quality correction and must produce a new
release tag rather than moving `v0.1`.

## Repository Layout

```text
README.md
release.json
manifest.csv
manifest.parquet
privacy/
  privacy_decisions.parquet
  face_blur_manifest.csv
labs/
  Araya/<original source path>
  Armnet/<original source path>
  ...
  UofT_TISL/<original source path>
```

`manifest.parquet` is the canonical one-row-per-video table. `manifest.csv`
contains the same 26,068 records for tools that do not read Parquet. Each row
records source repository, frozen revision, source path, release path, source
and release hashes, privacy status, and whether publication used a server copy
or local upload.

`privacy/privacy_decisions.parquet` explains the outcome for every video and
contains the Deface result, Qwen result when applicable, imported annotation,
final action, and exemption. It contains no frames, face boxes, or embeddings.

`privacy/face_blur_manifest.csv` contains only the 394 blurred videos. It
records whether the derivative was carried from a source repository or newly
rendered, original and blurred hashes, decision source, configuration hash,
and available tracking/render statistics.

## Preparing the Release

`publish_snapshot_release prepare` consumes the frozen scan, the four decision
manifests, and the three new blur-output manifests. It validates every local
overlay and every frozen Hub source hash before creating an action plan.

```bash
python -m oopsie_backend.scripts.publish_snapshot_release prepare \
  --scan-dir face_detection_results/oopsie-submissions-2026-08-22 \
  --annotation-manifest /path/to/2026-08-11/annotations.csv \
  --annotation-manifest /path/to/2026-08-24-qwen-positive/annotations.csv \
  --annotation-manifest /path/to/2026-08-24-qwen-reject-19/annotations.csv \
  --annotation-manifest /path/to/2026-08-24-qwen-reject-20/annotations.csv \
  --overlay-blur-manifest /path/to/qwen-positive/blurred_manifest.csv \
  --overlay-blur-manifest /path/to/qwen-reject-19/blurred_manifest.csv \
  --overlay-blur-manifest /path/to/qwen-reject-20/blurred_manifest.csv \
  --output-dir /path/to/oopsiedata-v0.1-release \
  --repo-id OopsieData-Submissions/oopsiedata-v0.1 \
  --release-version v0.1
```

This implementation is intentionally release-specific. Preparation refuses to
continue unless it resolves exactly 240 carried derivatives, 154 overlays, and
394 blurred videos. A future release should use a new frozen inventory and
explicit expected counts rather than silently reusing v0.1 decisions.

The generated `.internal/publish_actions.jsonl` is local resumable state and is
not uploaded. The user-facing CSV, Parquet, JSON, and README files contain no
local filesystem paths.

## Publishing Efficiently

Run publication in a detachable terminal:

```bash
tmux new-session -s oopsiedata-v01-publish

HF_XET_HIGH_PERFORMANCE=1 \
python -m oopsie_backend.scripts.publish_snapshot_release publish \
  --release-dir /path/to/oopsiedata-v0.1-release \
  --repo-id OopsieData-Submissions/oopsiedata-v0.1 \
  --expected-identity oopsiedata \
  --batch-size 75 \
  --upload-threads 8
```

The publisher creates or reuses a **private** destination and refuses to write
if it is public. For 25,914 unchanged videos it uses pinned cross-repository
`CommitOperationCopy` operations, avoiding local download and upload. Only the
154 new derivatives, about 1.22 GiB, are uploaded from local disk.

Publication is resumable. Each completed batch is recorded atomically in
`.internal/publication.json`; reruns skip completed batches and reconcile any
commit that succeeded before local state was written. HTTP 429 responses use
automatic backoff, including the Hub's 320-repository-commits-per-hour limit.

After all 358 batches complete, the publisher queries all 26,068 destination
paths, compares remote sizes and SHA-256 values, and only then creates `v0.1`.

## Source Repositories and Original Media

The snapshot contains only the privacy-processed release bytes. In particular,
the original versions of the 154 new overlays were never copied into this
repository's history.

The two blur groups have different source-repository states:

- The earlier 240 blurred files are also on the five affected source
  repositories' `main` branches. Their originals remain available through
  `pre-face-blur-2026-08-12` and the private raw archive.
- The 154 new blurred files currently exist in the v0.1 snapshot and local
  validated render outputs. Their source repositories still serve the original
  bytes on `main`, and the frozen source revision is recorded per row.

All 394 original hashes were verified at the `original_source_revision` listed
in `privacy/face_blur_manifest.csv`. Before promoting the 154 new derivatives
into source repositories, create an explicit pre-blur tag on each affected
repository and perform the same remote-hash validation used for the first 240.

## Downloading

Users with access can download the full immutable release:

```bash
hf download OopsieData-Submissions/oopsiedata-v0.1 \
  --repo-type dataset \
  --revision v0.1 \
  --local-dir ./oopsiedata-v0.1
```

Download a single lab without transferring the others:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OopsieData-Submissions/oopsiedata-v0.1",
    repo_type="dataset",
    revision="v0.1",
    allow_patterns=["labs/UofT_TISL/**"],
    local_dir="oopsiedata-v0.1",
)
```

## Local Retention

After remote validation, raw annotation media was removed from the four local
annotation workspaces. The CSV decisions, histories, render manifests, scan
records, validated blur outputs, and publication ledger were retained during
collaborator review. Raw inputs can be rematerialized from the pinned source
revisions; the local annotation copies are not authoritative storage.

If collaborator review requires a correction, update the snapshot's `main`
branch, regenerate affected metadata and hashes, validate the entire resulting
tree, and create a new immutable tag such as `v0.1.1`. Never move or overwrite
`v0.1`.
