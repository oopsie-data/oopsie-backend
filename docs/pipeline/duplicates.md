# Duplicates

Labs sometimes upload the same recording twice: re-exported after a fix, trimmed, or copied into a second repo. Some upload the same video file for two cameras or two episodes. The duplicate stages find both cases, so that each recording counts once and broken videos are caught.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `duplicate_episodes` | `preqa` | `episode_index` | `duplicate_of`, `duplicate_reasons`, `duplicate_group` |
| `duplicate_videos` | `preqa` | `duplicate_episodes` | `video_duplicates`, `has_duplicate_video` |

## Duplicate episodes

Two on-disk episodes are copies when any of three rules links them:

| Rule | Condition |
|---|---|
| `identical_data` | the same `content_hash`, i.e. identical action and state arrays |
| `identical_videos` | the same Hub hash for every camera's video |
| `same_metadata` | the same lab, timestamp, duration, instruction, robot and cameras, with a recorded timestamp; this catches re-exported or trimmed copies |

Copies linked by any rule form a group, and one copy of each group is kept: the first one from a contributor repo, then by repo and path. Every other copy gets `duplicate_of`, the key of the kept copy. `duplicate_reasons` lists the rules that link an episode to the others, and `duplicate_group` lists the other copies.

Every later stage skips episodes with `duplicate_of` set. Copies are never critical, since they are left out anyway.

## Duplicate videos

A video file that is byte-identical to another camera's or another episode's video is almost always an export error: the lab wrote one camera's file under two names. `video_duplicates` lists, for each such camera, the shared hash and every other episode and camera with the same file, and `has_duplicate_video` summarizes it. Videos are compared by their Hub hash, and copies marked `duplicate_of` are left out, since a copy shares its videos by definition.

An episode with a duplicate video is [critical](critical-episodes.md) by default.

## Running it by hand

```bash
uv run python -m oopsie_backend.preqa.duplicate_episodes --index episode_index.parquet
uv run python -m oopsie_backend.preqa.duplicate_videos --index episode_index.parquet
```
