# QA spot check report

Source: the 10 returned manifests in `qa_manifests/` (980 episodes).

## Coverage

| Bundle | Reviewer | Episodes | Notes "ok" | Notes blank | Flagged |
|---|---|---|---|---|---|
| 01 | Karen | 100 | 79 | 0 | 21 |
| 03 | Claas | 100 | 80 | 0 | 20 |
| 04 | rw435 | 100 | 0 | 86 | 14 |
| 05 | MarcelHussing | 100 | 44 | 0 | 56 |
| 06 | maxrudolph | 100 | 53 | 0 | 47 |
| 07 | Carl | 100 | 92 | 0 | 8 |
| 08 | siddhant | 100 | 70 | 0 | 30 |
| 09 | Maria | 100 | 87 | 0 | 13 |
| 10 | zhouzypaul | 100 | 0 | 61 | 39 |
| 11 | Yifan | 80 | 65 | 0 | 15 |

- **Bundle 02 is missing.**
- **Blank notes:** the reviewers of bundles 04 and 10 left notes empty rather than writing "ok". This probably means "ok", but it is unconfirmed.
- **Reviewer strictness varies**, from 8 flags (Carl) to 56 (MarcelHussing). Flag counts therefore can't be compared directly across bundles.
- **Reviewer fields are consistent:** every episode's `qa_reviewer` matches its bundle's reviewer, and no episode appears twice.

## Clean labs (no flags)

Araya, Berkeley_HybridRobotics, GT_PAIR, GT_RL2, INRIA_SC, Princeton_PRISM, UNC_MD, WUSTL_MIG.

## Systematic problems at the lab level

| Lab | Flagged | Issue |
|---|---|---|
| MIT_Improbable | 20/20 | Global camera is rotated. The failure notes are confusing, and several videos are cut short, so the success/failure label can't be checked. |
| VUA_KL (`lab_id` UA_KL + VUA_KL) | 13/20 | Camera labels are swapped; one video is choppy. |
| NUS_CLeAR | 18/20 | Descriptions look generated ("Push 4 of 7 in an object-centric push-planner run…"). Instructions contain "watch the video carefully" and an undefined "goal pose", and objects are named wrong. |
| Stanford_JW / Stanford_SIALab | 10/20 each | Failures have no failure annotation (Stanford_MSL has 1 such case). |
| KAIST_KL | 17/20 | Low resolution throughout; humans in the frame; vague instructions ("designated position", "PET" written as "pet"). |
| BONN_MB (incl. 3 episodes with `lab_id` "123") | 13/17 + 2/3 | The "bin" from the instruction is not visible in any camera view. |
| ETH_RK | 11/20 | Descriptions contain unrelated info (8); poor camera placement (3). |
| UTA_RPL | 8/20 | Long frozen stretches at the start, one completely dark video, one 1-second video, and a hand blocking the cameras. |
| KAIST_JS | 9/20 | Unclear success criterion (labelled failures look like successes); "dolls" seems to be the wrong noun. |
| UdeM_REAL / MILA | 13/20 | Low resolution; the robot barely moves; success labels should be "suboptimal" or "side effects"; choppy at action-chunk boundaries. |
| Charles_UFAL | 4/20 | Videos appear sped up. |
| CMU_RPAD | 4/20 | Video artifacts or desync; 2 labels should be "success suboptimal". |
| UW_REAL | 7/20 | Wrist camera videos don't load (2); low resolution; fisheye camera; the "carbon fiber layup" task is unclear. |
| KIT_ALR | 6/20 | Left/right in the instruction is ambiguous in the wrist view. |
| Northstar | 9/20 | Instructions wrong (purple vs. blue objects, no purple crate) or too vague ("red cube demo"); one episode has no failure annotation. |

## Problems across labs

- **Privacy:** MIT_CLEAR has 2 episodes with unblurred faces in the wrist camera. Bystanders appear in NYU_Clark, NYU_AI4CE, KAIST_KL and Stanford_FFL; in Stanford_FFL a hand touches the object.
- **Videos cut early or with idle padding:** NYU_AI4CE, PSU_FMC, USC_Lira, Berkeley_RAIL and UofT_TISL.
- **Corrupted frames:** Stanford_FFL (2).
- **Instruction doesn't match the scene:**
  - Armnet: there is no pink lamp.
  - Harvard_EML: one episode's language belongs to a different episode.
  - UofT_RVL: there is no bucket.
  - PSU_FMC: there are no vegetables.
  - TUD_JP: the cup is dropped.
- **Success/failure label disputed:** KAIST_JS, MIT_Improbable, Princeton_IRoM, RIKEN_AIP (2), SE3Labs, Toloka, UIUC_Reid, UofT_RVL, Harvard_EML and MIT_CLEAR.
- **Failure category or severity disputed:**
  - UCSD_ARCLAB (7): episodes are labelled "collision" or "medium" severity where the reviewer would say manipulation or low severity.
  - UIUC_DASLAB (2): labelled "sequencing_semantic" where the reviewer would say grasp.
  - Princeton_PRPL and Toloka: one each.

## Metadata inconsistencies

In six cases the repo name and the `lab_id` field disagree:

| Repo | `lab_id` | Episodes |
|---|---|---|
| Enatic_OpenArm | Enactic_OpenArm | 20 |
| UNC_MD | UNC_IDEAL | 20 |
| AutoEval_origin | Berkeley_RAIL | 20 |
| VUA_KL | UA_KL | 6 |
| UdeM_REAL | MILA | 3 |
| BONN_MB | 123 | 3 |

## Automated video checks

The video checks of the action QA (`docs/video-checks.md`) were run on the 200 episodes of bundles 01 and 03, the bundles available locally.

| Episode | Camera | Finding | Reviewer note |
|---|---|---|---|
| Enatic_OpenArm 03-075 (`77.h5`) | wrist_left, wrist_right | Hard cuts at 5.0 s and 10.9 s on both wrist cameras; footage is missing. Confirmed visually. | ok |
| CMU_RPAD 03-011 | back | 23 identical frames (1.5 s) while the gripper moves. Confirmed visually. | ok |
| CMU_RPAD 03-016 | front | 24% duplicate frames while the robot moves (choppy) | "video artifacts, desync" |
| Araya, 9 episodes | all | Irregular frame timestamps (frame interval varies by 12–35% of its mean) | ok |
| ETH_RK 03-046 | oakd_side_view | 22% duplicate frames while the robot moves; not verified visually | ok |
| ETH_RK 03-060 | oakd_side_view | Image jump at 8.4 s; false positive (an object enters view between two frames) | ok |

- **Low resolution:** every camera of Araya, AutoEval_origin (256×256) and Berkeley_HybridRobotics is flagged, as are some cameras of CMU_RPAD (320×240) and ETH_RK (224×224). Reviewers raised resolution only for Berkeley_RAIL (AutoEval_origin).
- **No findings for decoding or brightness** in either bundle.
- **Not detected:**
  - The Charles_UFAL "sped up" videos have consistent fps, frame counts and trajectory lengths, so the speed check cannot catch them.
  - Problems in bundles that are not available locally (Stanford_FFL corrupted frames, UTA_RPL frozen and dark videos, VUA_KL choppy video) remain untested.

## Follow-ups

1. Get the bundle 02 manifest back from its reviewer.
2. Contact the labs with systematic issues: MIT_Improbable, VUA_KL, NUS_CLeAR, Stanford_JW, Stanford_SIALab, BONN_MB, ETH_RK and UTA_RPL.
3. Check MIT_CLEAR for unblurred faces before release.
4. Decide whether disputed category/severity calls (e.g. UCSD_ARCLAB) should change the labels or only be recorded.
