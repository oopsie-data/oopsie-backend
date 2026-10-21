# OopsieData Lab Data-Quality Exceptions

Oct 4, 2026

## Checks performed

The per-episode action and video checks are documented in [docs/action-checks.md](../docs/action-checks.md) and [docs/video-checks.md](../docs/video-checks.md). The stats pipeline additionally performs these checks:

| Category | Check | Status | What it flags |
| --- | --- | --- | --- |
| General | Implausible timestamp | Implemented | Timestamp decodes to before 2000 (e.g. a monotonic clock bug) → stamped with a placeholder date instead of a bogus 1970 date |
| General | Placeholder operator name | Implemented | Generic/blank `operator_name` (`pending_operator`, `operator`, blank...) → excluded from per-individual requirements, flagged separately |
| Action | Tag-sum vs. episode count | Implemented | Per-dataset count of each action tag vs. episode count — catches labs recording multiple simultaneous representations |
| Action | Duplicate action arrays | Implemented | Two action keys on the same episode hold identical values → likely copy-paste bug |
| Action vs. Observations | Scale consistency | Implemented | Action's magnitude is wildly out of proportion to the robot's own state (wrong units, bad normalization, corrupt data) |
| Action vs. Observations | Tracking residual | Implemented | Action doesn't correspond to where the robot's state actually ends up next step (misaligned data, wrong field) |
| Action vs. Observations | Delta-action detection | Implemented | Action looks like *state\[t+1\] - state\[t\]* (a delta) rather than the absolute state we require |
| Action vs. Observations | Action copies state | Implemented | Joint/cartesian action equals the recorded state at the same step (lag 0 only; matching the next step's state is not flagged) in ≥90% of steps → actions were rebuilt from states afterwards, not logged commands |
| General | Placeholder metadata | Implemented | `operator_name`, `policy_name`, `robot_name` or `gripper_name` is blank, a placeholder (`unknown`, `test`, `pending_operator`...), an unfilled template (`your_...`), or - for operators - a program/lab rather than a person (`SOAR_policy`, `autoeval_orchestrator`, `cmu_aire_lab`) |
| Action vs. Observations | Required state keys | Implemented | A state the action space needs is missing or empty: `gripper_position` always, `joint_position` for joint actions, `cartesian_position` for cartesian actions (same rule oopsie-data-tools enforces on the robot profile since Jul 28 2026; older or converter-written data can lack it) |
| General | Action chunk shape | Implemented | An arm action (`joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`) is shaped *(T, chunk, dof)* instead of *(T,)* or *(T, dof)* → the policy's whole action chunk was saved at every step instead of the one action executed |
| General | Constant action | Implemented | An arm action never changes over the whole episode → a stuck/failed policy, or a placeholder action written by a converter; reports the value and how far the matching state moved |
| General | Command rate | Implemented | Each new arm action is held for the same N ≥ 2 steps in ≥ 80% of holds → commands arrive at `control_freq` / N, not at the declared `control_freq`; reports N and the effective rate |
| Action vs. Observations | Length mismatch | Implemented | An action and the state it's observed through have different numbers of timesteps |
| joint\_position | Joint names | Implemented | Profile's action/state joint names missing, blank/duplicated, not matching each other, or out of order (numbering, or base-to-wrist chain - e.g. an alphabetical ROS list) |
| joint\_position | DOF | Implemented | Number of joint values doesn't match the number of joint names |
| joint\_position | Radians | Implemented | Joint values beyond 6.5 → degrees or a -100..100 range; reports a best-guess unit |
| joint\_position | Step jump | Implemented | A joint moves faster than 6.28 rad/s between two steps at the profile's `control_freq` |
| joint\_position | Gripper in joints | Implemented | A joint name contains "gripper" → gripper value mixed into the joint vector |
| joint\_position | Delta-action detection | Implemented | Joint action looks like a delta instead of an absolute position (joints paired with the state by name) |
| joint\_position | Tracking | Implemented | Median per-joint error between action\[t\] and state\[t+1\] above 0.2 rad |
| joint\_velocity | DOF | Implemented | Number of joint velocities doesn't match the number of joint names |
| joint\_velocity | Speed limit | Implemented | A joint commanded faster than 3.14 rad/s |
| joint\_velocity | Normalized range | Implemented | Velocities peak at exactly ±1 → normalized (DROID-style), not rad/s |
| joint\_velocity | Integration | Implemented | velocity / control\_freq vs. the joint state's per-step change: ratio outside 0.25-8 (suspected deg/s or per-step deltas), correlation below 0.3, or an exact copy of the state's motion |
| joint\_velocity | Looks like position | Implemented | The "velocity" sits on top of the joint position state → saved under the wrong key |
| joint\_velocity | Smoothness | Implemented | More than 30% of steps reverse direction |
| cartesian\_velocity | Shape | Implemented | Not 6 values per arm (3 linear + 3 angular) |
| cartesian\_velocity | Speed limit | Implemented | Linear above 2 m/s or angular above 6.28 rad/s |
| cartesian\_velocity | Normalized range | Implemented | Velocities peak at exactly ±1 → normalized, not physical units |
| cartesian\_velocity | Integration | Implemented | Same as joint\_velocity, separately for linear (vs. xyz change) and angular (vs. rotation change); reports whether base or end-effector frame fits |
| cartesian\_velocity | Linear/angular order | Implemented | Swapping the two halves matches the real motion clearly better |
| cartesian\_velocity | Looks like position | Implemented | The linear part sits on top of the state's xyz |
| cartesian\_velocity | Smoothness | Implemented | More than 30% of steps reverse direction |
| cartesian\_position | Quaternion shape | Implemented | Dim isn't 7 (14 biarm), a quaternion isn't unit-norm (an optional scalar-first guess exists but is off: it misfires on gripper-down poses) |
| cartesian\_position | Position units | Implemented | Action or state xyz beyond 3 m, or action xyz ≫ state xyz → not in metres (mm/cm) or wrong frame; reports a best-guess unit (`cm`/`mm`) |
| cartesian\_position | Quaternion sign flips | Implemented | Consecutive quaternions with dot < 0 — same orientation written as -q, which breaks learning on raw values |
| cartesian\_position | Rotation jumps | Implemented | Rotation between consecutive steps faster than 360°/s at the profile's `control_freq` |
| cartesian\_position | Rotation tracking | Implemented | Median angle between action rotation and the next state's rotation above 20° → wrong frame/convention |
| cartesian\_position | Delta-action detection (xyz) | Implemented | Action xyz looks like a delta instead of an absolute position |
| cartesian\_position | Biarm ordering | Implemented | Action xyz matches the state better with left/right arms swapped |
| Video | Resolution | Implemented | A camera's short side is below 360 px |
| Video | Decoding | Implemented | Video file missing, ffmpeg decode errors, nothing decodes, or decoded frame count differs from the container's |
| Video | Brightness | Implemented | Median frame brightness too dark or too bright, or ≥ 10% near-black / near-white frames |
| Video | FPS | Implemented | Video fps ≠ `control_freq` (header value → wrong playback speed; frames still align if the count matches T) |
| Video | Duration | Implemented | Video duration ≠ trajectory duration → frame count doesn't match the number of steps, so video and actions are out of step |
| Video | Camera duration mismatch | Implemented | Cameras of one episode differ in duration |
| Video | Corrupted frames | Implemented | A single frame unlike both its neighbours |
| Video | Video jump | Implemented | A sudden image change (cut) while the robot state changes smoothly |
| Video | Choppy | Implemented | ≥ 20% duplicate frames while the robot moves |
| Video | Frozen | Implemented | ≥ 1 s of static video while the robot moves |
| Video | Recording gap | Implemented | Robot state and video jump at the same moment (state step ≥ 10x the surrounding second's average, video change ≥ 5x) → time is missing from the recording |
| Video | State jump | Implemented | Robot state jumps in one step while the video doesn't → a state glitch, or state and video out of step |

Note: the alpha/beta per-individual lab requirement pass/fail system is a separate compliance mechanism, not a data-integrity check, and isn't listed here.

## Purpose

This doc tracks data-quality findings surfaced while building OopsieData's internal stats dashboard (`internal_stats.html`) — things that look like submission-pipeline bugs or ambiguous conventions in specific labs' data, not bugs in the stats code itself. Use it to:

- Brief stakeholders in meetings on known data caveats before numbers get over-interpreted.
- Have ready-made, specific questions on hand when reaching out to a lab for clarification.
- Track status (open / asked / resolved) as labs respond.

Each section below names the lab(s), the exact numbers affected, and a suggested question. Last refreshed against data synced through 2026-10-04.

## Operator identity issues

Three labs' episodes are credited to placeholder or generic operator names instead of a real person, so they can't be evaluated against the per-individual alpha/beta requirement (shown as a gray "no individual data yet" dot in the All Datasets table) even though the lab has submitted real episodes.

| Lab | Episodes affected | Name used | Suggested question |
| --- | --- | --- | --- |
| Harvard\_EML | 814 (100% of submissions) | `pending_operator` | Who actually ran these sessions? Can future submissions fill in the real operator name instead of the recorder template's default? |
| Stanford\_JW | 914 (100% of submissions) | `operator` | Same — generic placeholder, not a real identity. |
| UofT\_RVL | 62 | *(blank)* | A subset of episodes has no operator name at all — were these recorded by a different pipeline than the rest of UofT\_RVL's submissions? |

**Status: open — not yet contacted.**

## Timestamp data quality — Harvard\_EML

Harvard\_EML's episode `timestamp` HDF5 attribute holds a small number (e.g. `492806.19`) consistent with a relative/monotonic clock (`time.monotonic()`), not wall-clock Unix time. Read literally, this placed all 814 of their episodes on dates in January 1970.

**Mitigation already applied** (no lab-side action needed to unblock reporting): the stats pipeline now treats any timestamp that decodes to before 2000-01-01 as implausible and stamps it with a placeholder date (2026-09-01) instead, so the public/internal timelines no longer show the bogus 1970 cluster. This is a stopgap, not a fix — Harvard\_EML's true submission dates are still unknown to us.

**Suggested question:** Does your episode recorder write `time.monotonic()` (or similar) into the `timestamp` field instead of `time.time()`? If so, can it be corrected for future submissions so the real submission date is recoverable?

**Status: mitigated in reporting — lab not yet contacted about the source.**

## Lab ID / dataset mismatch — Enatic\_OpenArm vs Enactic\_OpenArm

The HF dataset repo is `OopsieData-Submissions/Enatic_OpenArm`, but some episodes inside it carry `lab_id = "Enactic_OpenArm"` (extra "c"). This creates a phantom near-duplicate entry in per-lab breakdowns — the real episode counts land under `Enactic_OpenArm`, while `Enatic_OpenArm` (guessed from the repo name) shows up separately as an empty "no data yet" entry.

**Suggested question:** Which spelling is correct — the repo name or the `lab_id` attribute — and can future submissions use one consistent spelling?

**Status: open — not yet contacted.**

## Action space double-counting

The internal dashboard's "Arm action space" chart counts one tag per matching `action_space` entry per episode. For most labs that equals the episode count exactly (as the Gripper chart does, 0 discrepancy). Six labs instead record more than one arm-type representation in the same episode (e.g. both `joint_position` and `cartesian_position`), so their episodes get counted twice:

| Lab | Excess count | Representations recorded together | Episodes affected |
| --- | --- | --- | --- |
| Princeton\_PRPL | 2,413 | `joint_position` + `joint_velocity` | 2,413 of 3,813 (mixed — some episodes have only one) |
| McGill\_DM\_HL | 1,029 | `cartesian_position` + `joint_position` | all 1,029 |
| GT\_RL2 | 150 | `cartesian_position` + `joint_position` | 150 of 871 (mixed) |
| VUA\_KL | 72 | `cartesian_position` + `cartesian_velocity` + `joint_position` + `joint_velocity` (4-way) | 24 episodes, 3 excess each |
| Princeton\_VLL | 54 | `cartesian_position` + `joint_position` | all 54 |
| WUSTL\_MIG | 28 | `cartesian_position` + `joint_position` | all 28 |
| **Total excess** | **3,746** |  | matches the observed Arm-chart discrepancy exactly |

This likely isn't an error — some converters intentionally provide the same action in multiple coordinate frames. But Princeton\_PRPL and GT\_RL2 are inconsistent *within themselves* (only some of their own episodes double up), which looks more like an unintentional converter quirk than a deliberate per-episode choice.

**Suggested question:** Is recording multiple arm action representations per episode intentional? If so, is there a "primary" representation we should treat as canonical for a clean per-episode breakdown?

**Status: open — not yet contacted.**

## Action chunk checks

Chunked policies (ACT, pi0, diffusion policy, ...) predict several future actions per query. Every recorded step must hold only the action executed at that step. Checked 2026-10-10 against all 188,307 submitted episodes (shapes) and a random sample of 25 episodes per lab (values).

**Implemented: action chunk shape** (`general/action_chunk_shape`, arm action spaces only). An arm action stored as *(T, chunk, dof)* means the whole chunk was saved at every step. Its first axis still matches T and its last still matches the DOF, so the length and DOF checks miss it. **No episode in the current data has this.**

**Considered and not added: value-based chunk checks.** A chunk logged wrongly with a valid per-step shape shows up as the same action repeated for several steps. The sample has three such patterns. Each was looked at as a possible chunk check, and each has an explanation where the recording is correct:

| Pattern | Seen in | Why it isn't a chunk bug |
| --- | --- | --- |
| Holds in multiples of the chunk size (8 steps) | roboarena\_upload, `paligemma_binning_droid` | The held value is the binning policy's "no motion" bin (≈0.01 rad/s), and the joints really are still during it. Its 8 actions differ only when the arm moves. No hidden actions were executed. 3 of these episodes never move at all, which **Constant action** now catches. |
| A short hold once per chunk (5–9 steps every 24–35) | AI2\_MOLMOACT2, KIT\_ALR, UIUC\_Reid | The robot keeps its last target while the next chunk is inferred. That's what it was actually commanded, so the data is accurate. The pause itself is what the video `periodic_skips` check covers. (Inferred from how regular it is; not confirmed with the labs.) |
| Every action held for the same N steps | Stanford\_JW (N = 3 at 30 Hz), NUS\_CLeAR, Charles\_UFAL | Commands are sent at a lower rate than the recording rate. This isn't a chunk issue but a `control_freq` mismatch, which **Command rate** now catches. |

A real chunk-logging bug would need the action held for N steps while the robot moves by a meaningful amount, *and* motion during the hold that doesn't match the held command. None was found, so the check isn't worth its false alarms yet. Revisit if a lab's inference script shows the bug. A chunk flattened into one *(chunk × dof)* vector also wasn't found: no action width is a multiple of its joint count or of the matching state's width.

**Sample findings from the two new checks** (25 episodes per lab, not yet the full data):

| Check | Lab | Flagged | Detail |
| --- | --- | --- | --- |
| Command rate | Stanford\_JW | 17/25 | `cartesian_position` held 3 steps → 10 Hz commands vs. declared 30 Hz |
| Command rate | NUS\_CLeAR | 3/25 | `cartesian_position` held 2 steps → 10 Hz vs. 20 Hz |
| Command rate | Charles\_UFAL | 1/25 | `joint_position` held 6 steps → 5 Hz vs. 30 Hz |
| Constant action | roboarena\_upload | 3/25 | `joint_velocity` constant at ≈0.01 rad/s; arm moves ~0.001 rad (policy never moved) |
| Constant action | VUA\_KL | 2/25 | `joint_position` fixed at the home pose while the joints move 1.8 rad → placeholder action |
| Constant action | AutoEval\_origin | 1/25 | `cartesian_position` all zeros + identity quaternion, state also zero → placeholder data |

**Status: checks implemented; findings from a sample only — rerun `run_action_qa.py` on the full data before contacting labs.**

## Labs to contact — summary

| Lab | Issue | Status |
| --- | --- | --- |
| Harvard\_EML | Placeholder operator name + monotonic timestamp | Open |
| Stanford\_JW | Placeholder operator name | Open |
| UofT\_RVL | Blank operator name (62 episodes) | Open |
| Enatic\_OpenArm / Enactic\_OpenArm | lab\_id spelling mismatch | Open |
| Princeton\_PRPL | Arm action space double-counted (mixed within lab) | Open |
| McGill\_DM\_HL | Arm action space double-counted (all episodes) | Open |
| GT\_RL2 | Arm action space double-counted (mixed within lab) | Open |
| VUA\_KL | Arm action space double-counted (4-way) | Open |
| Princeton\_VLL | Arm action space double-counted (all episodes) | Open |
| WUSTL\_MIG | Arm action space double-counted (all episodes) | Open |

Update the Status column as each lab responds or the issue is resolved.
