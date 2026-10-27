# Video checks

The video checks run as the `video` category of the action QA. Their results land in the same report and dashboard (`reports/action_qa.html`) as the action checks: one grid column per check and a per-episode drilldown keyed by camera.

```bash
uv run python -m oopsie_backend.qa.run \
  --data-dir /corral/utexas/IRI26024/oopsiedata --limit 50 --workers 16
```

Each camera video is decoded in full once (about 2 CPU-seconds per episode), so `--workers` (`run.workers`) matters for large runs. Toggles and thresholds live in the `video` section of `configs/action_qa_config.yaml`; set `video.enabled: false` to run the action checks alone.

Frame differences are mean absolute differences of 96x96 grayscale frames (0–255).

| Check | Flags when |
|---|---|
| `resolution` | short side < 360 px |
| `decoding` | file missing, ffmpeg errors, no frames decoded, or decoded frame count differs from the container's by more than 1 |
| `brightness` | median luma < 30 or > 225, or ≥ 10% of frames with luma < 15 or > 240 |
| `fps` | fps differs from `control_freq` by > 5% (a header value: playback speed is wrong, but frames still align if the frame count equals T) |
| `duration` | video duration (decoded frames / fps) differs from `T / control_freq` by more than max(0.5 s, 5%) — with fps = `control_freq`, the frame count doesn't match T, so frame *i* is no longer step *i* |
| `camera_duration_mismatch` | cameras of one episode differ in duration by > 0.5 s (keyed `all cameras`; skipped with one camera) |
| `corrupted_frames` | A frame differing ≥ 20 from both neighbours while the neighbours differ ≥ 4x less from each other |
| `video_jump` | A frame difference ≥ 40 and ≥ 20x the median of the surrounding second, while the robot state doesn't jump within one step (state step < 5x its surrounding second's average); otherwise it's a recording gap |
| `choppy` | ≥ 20% duplicate frames (difference < 0.01) while the robot moves |
| `frozen` | ≥ 1 s of static frames (difference < 0.05) while the robot moves |
| `periodic_skips` | ≥ 6 isolated robot-state steps of ≥ 3x the local median step at a regular period (period score ≥ 0.4), in ≥ 60% of the periods while the robot moves, not explained by a position command that changed ≥ 3x as much as the state; ≥ 40% of them must be confirmed by a joint jump ≥ 0.015 rad (or end-effector jump ≥ 1 cm without joint state) beyond a normal step and a frame difference ≥ 2x the local median in some camera (keyed `all cameras`; skipped without joint or cartesian position state) |
| `recording_gap` | A robot-state step ≥ 10x the mean step of the surrounding second and ≥ the episode's 95th-percentile step, with no jump on either neighbour, while most of the surrounding second is moving (at least half its steps pass the "robot moves" test), **and** a camera's frame change within one frame of it ≥ 5x the surrounding second's (keyed `all cameras`) |
| `state_jump` | The same robot-state jump, but no camera jumps at that moment: a state glitch, or state and video out of step (keyed `all cameras`) |

"Robot moves" means the non-gripper robot state changes by more than 10% of the episode's 95th-percentile step. Without robot state, frozen, choppy, `periodic_skips`, `recording_gap` and `state_jump` are skipped (skip reason `no_robot_state`); episodes without video files skip all video checks (`no_video_files`).

`periodic_skips` catches recordings that pause at a regular period, typically during each action-chunk inference, while the low-level controller keeps the robot moving. The first sample after the pause then shows a large jump in the robot position, and the cameras skip the same stretch of time. A policy whose first step after each replan commands a larger change than usual also produces periodic state jumps. There the logged command jumps further than the state, which then follows it, so these jumps are dropped. In a recording pause the command log skips ahead with the state, so the state jumps about as far as the command. A held command (unchanged over several steps while the controller executes it) explains no jump.

The `fps` and `duration` checks cannot detect videos that are sped up consistently with the robot state, i.e. a `control_freq` that is wrong for both.
