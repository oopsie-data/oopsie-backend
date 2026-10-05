# OopsieData Lab Data-Quality Exceptions

Oct 4, 2026

## Checks performed (methodology)

### General QA

1. **Implausible timestamp detection** *(implemented)* — any episode timestamp that decodes to before 2000-01-01 (e.g. a monotonic/relative clock mistakenly used instead of wall-clock time) is flagged and replaced with a placeholder date, rather than silently appearing as a bogus 1970s date on the dashboard. (Caught Harvard\_EML.)
2. **Placeholder/generic operator-name detection** *(implemented)* — episode `operator_name` values matching a configurable list of placeholders (`pending_operator`, `operator`, blank, etc.) are excluded from per-individual lab-requirement evaluation and separately flagged with a count + the names used, so a lab can't pass requirements via an unidentifiable "individual." (Caught Harvard\_EML, Stanford\_JW, UofT\_RVL.)

### Action QA

1. **Action-space tag-sum vs. episode-count check** *(implemented)* — per dataset, the summed counts of each arm-type action tag (`joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`) and each gripper-type tag are compared against the dataset's episode count. Multiple simultaneous action representations are allowed, so "more than episode count" is expected, but it surfaces exactly which labs record multiple representations and by how much. (Found 6 labs with excess arm-tag counts totaling 3,746 extra tags.)
2. **Duplicate/copy-pasted action array check** *(implemented)* — for episodes that log 2+ action representations, compares every same-shaped pair of arrays for exact/near-exact equality; flags a likely copy-paste bug (recording one quantity twice under different names) rather than two genuinely distinct representations.

### Action vs. Observations QA

1. **Scale consistency check** *(implemented)* — compares the scale (spread) of recorded action values to the robot's own observed state values; flags episodes where the action's magnitude is wildly out of proportion to the robot's real motion (e.g. wrong units, a normalization bug, or corrupted data).
2. **Tracking residual check** *(implemented)* — checks whether each action's value is physically close to where the robot's state actually ends up one timestep later; flags episodes where the action and the robot's real motion don't correspond at all (misaligned data, wrong field, garbage values).
3. **Delta-action detection** *(implemented)* — labs were told to submit absolute actions, not deltas (see `oopsie-docs/docs/format.md` / `data-collection.md`). For each timestep, computes how well the action matches "the next absolute state" vs. how well it matches "the change in state from this step to the next," and flags episodes/datasets where the delta explanation fits much better — i.e. where a lab appears to be submitting delta actions despite being asked not to.

*Note: the alpha/beta per-individual lab requirement pass/fail system is a separate compliance-grading mechanism (not a data-integrity check) and is intentionally not listed here.*

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
