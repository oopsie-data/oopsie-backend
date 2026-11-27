# Dataset analysis

The dataset analysis computes the statistics for the Oopsie-Data paper, as planned in `plans/stats.md`: what tasks, robots and failures the dataset covers, how well annotators agree, and how failures show in the robot's motion. It writes them as JSON reports, one per input group. Entry numbers in the reports (`15_failure_types`) are the entry numbers of the plan. Every group covers contributed episodes only (`repo_kind == contributor`) that are on disk, readable, not a duplicate copy, not critical and not excluded by the [annotation flags](annotation/flags.md) (`annotation_excluded`).

| Group | Input | Entries |
|---|---|---|
| `index` | the episode index | 1–3, 5–7, 9, 10, 12, 46; 15, 18, 21–25, 28 from the contributors' labels |
| `annotations` | the annotators' returns with failure segments, from the index's `annotation_*` columns | 14, 17, 19, 27, 29–33, 37, 42–45, 47; 15, 18, 21–25, 28 from the annotators' labels |
| `data` | action features read from the episode files | 36, 38–41 |

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `dataset_analysis` (manual) | `analysis` | `critical_episodes`, `annotation_flags`, the unification stages, `analysis_config` | `<reports_dir>/analysis/index.json` |
| `annotation_analysis` (manual) | `analysis` | the same | `<reports_dir>/analysis/annotations_<batch>.json` per batch, `annotations.json` over all batches |

Both stages are manual because the nightly job does not install the `analysis` extra. The `data` group reads every episode file and runs only as a command.

All groups need the index columns of the embodiment, policy and camera name stages. Settings, such as the embedding model, the similarity thresholds, the verb synonyms and the feature thresholds, are in `configs/analysis.yaml`.

## Installation

The language entries and the failure prediction need the `analysis` extra, which adds spaCy with its English model, sentence-transformers and scikit-learn:

```bash
uv sync --extra analysis
```

The embedding model is downloaded on first use into the Hugging Face cache (`HF_HOME`). `--no-language` computes the `index` and `annotations` groups without the extra, leaving out entries 1–3, 25, 47 and the verbs of entry 5.

## Commands

```bash
uv run oopsie stats analysis index --index episode_index.parquet --out reports/analysis/index.json
uv run oopsie stats analysis annotations --index episode_index.parquet [--batch eval|train] --out reports/analysis/annotations.json
uv run oopsie stats analysis features --index episode_index.parquet --data-dir <mirror> --out <features.parquet> [--segments]
uv run oopsie stats analysis data --index episode_index.parquet --features <features.parquet> --out reports/analysis/data.json
uv run oopsie stats analysis render reports/analysis/*.json
```

`render` writes a Markdown version of each report next to it (`index.json` → `index.md`), with one section per entry and tables for distributions, summaries and group comparisons. Long tables show their first 25 rows, and group comparisons show the 12 most frequent values; the JSON keeps everything.

## Language

Instructions are lowercased and parsed with spaCy. Only the first sentence counts, since later sentences of long instructions describe the setup or the success condition. Each verb that heads the sentence or is coordinated with another step verb starts a step, so "pick up the cup and place it in the sink, then close the drawer" has the steps `pick`, `place` and `close`.

- A step's verb is its lemma with particle, merged through `verb_synonyms` ("grab", "pick up" → `pick`). A light verb (`light_verbs`, e.g. "use the box to push the cube") is replaced by the verb of its purpose clause.
- A step's objects are the lemmatized head nouns of its direct and prepositional objects, without spatial nouns (`object_stopwords`).

Word clouds count the lemmatized nouns, verbs and adjectives of a text once per text, without stop words and the words of `word_stopwords` (generic words such as robot or attempt, sides and colors). Entry 2 counts the first sentence of each episode's instruction, so every episode counts once; entry 47 counts the annotators' failure descriptions, which are the episode descriptions of labels with a failure outcome and the labels of all segments. Its `verbs` keep only verbs and adjectives, which name what went wrong rather than the objects involved.

Task clusters merge instructions whose sentence embeddings reach a cosine similarity threshold: they are the connected components of the graph that links every such pair. Entry 1 reports the number of clusters and the effective number of tasks, `exp(entropy)` of the episodes over the clusters, for each threshold in `similarity_thresholds`. Entry 5 counts clusters at `similarity_threshold`.

## Episode length

Entry 6 summarizes `duration_seconds` overall, per policy class, per contributor outcome and per failure category. `outcome_histogram` also counts the episodes of each outcome in 40 log-spaced bins from 1 s to 1,000 s; shorter and longer episodes fall into the end bins.

## Composition hours

Entry 46 sums `duration_seconds` into episodes and hours per verb, robot arm (`unified_arm`), end effector type, policy class and contributor outcome. In `verbs` an episode counts once for each of its distinct verbs; `verb_sets` counts it once under all its verbs joined by `+`, so that verbs can be grouped without counting an episode twice.

## Labels

The contributors' labels are each episode's primary annotation, the latest with a timestamp. The annotators' labels have one row per annotator and episode; an annotator's categories of an episode are the union of the categories of their segments. Category statistics use failed labellings. A labelling with several categories counts once for each, and one without any counts as `uncategorized`. Entry 18 also counts every pair of categories a failed labelling gives together (`grasp + manipulation`), as a share of failed labellings. Shares come with 95% Wilson intervals, and groups smaller than `min_group_episodes` are left out of comparisons.

Contributors mostly upload balanced sets of successes and failures, so the reports describe distributions within the failures, and overall outcome shares describe the uploaded data, not failure rates.

## Annotator returns

The `annotations` group and the segment windows of the `features` group read the collected annotations from the index's `annotation_*` columns ([index](../reference/index-columns.md#annotation)), which `oopsie annotation pipeline index` writes. Both kinds of bundle count unless `--batch` names `eval` or `train`. Each annotator's latest record of an episode counts. Records whose watched videos differ from the episode's current ones are left out, since the episode changed after it was annotated. So are the records the annotation flags mark unusable (`annotation_unusable`), such as those of an annotator who could not watch the video. The index keeps all of these records.

Outcomes are `failure`, `success`, `success_suboptimal`, `success_side_effect` and `incomplete` (the robot acted correctly but the episode ended before the task was done).

Entry 17 counts segments per label and per episode; an episode's count is the lower median of its annotators' counts, so an episode counts k failures only when at least half of its annotators mark k or more segments.

Entry 14 divides the start of an annotator's first segment by the episode duration and also reports it as a histogram of 20 equal bins over 0–1.

Segment bounds are inclusive frame indices in the index and are converted to seconds, `start / control_freq` and `(end + 1) / control_freq`. A segment with several categories is paired once, with its categories joined by `+`.

## Agreement

Segments of two annotators of an episode are paired greedily by their overlap (IoU), highest first; unpaired segments count as disagreement on the number of segments. Agreement is Krippendorff's α: nominal for outcomes and categories, ordinal for severity. Annotators are anonymized (`A1`, `A2`, …) in the bias report and the list of changed instructions.

Entry 30 reports the IoU of paired segments also as a histogram of 20 equal bins over 0–1, and the share of paired segments whose starts are within 0.5, 1, 2, 5 and 10 seconds of each other. Paired segments ignore annotator pairs where only one annotator marks segments. Entry 30 therefore also reports the IoU of the time two annotators' segments cover, with such pairs counted as 0, and for episodes with three annotators whether all three cover a common moment.

"Failure or not" compares `outcome == failure` against every other outcome. Entry 27 compares the contributor's label with every annotator label, with the annotators' majority (episodes with a tie are left out) and with unanimous annotators; entry 29 compares annotators with each other.

The per-episode agreement score of entries 43 and 44 is the mean over annotator pairs of three parts, each between 0 and 1: the outcome (1 for the same outcome, 0.5 when both or neither are a failure), the Jaccard index of the two category sets, and the IoU of the time their segments cover. Two labellings without categories or segments agree fully on that part. Episodes that every annotator calls a success without segments score 1, so the most agreed episodes are listed among those that every annotator calls a failure and that have segments.

## Action features

`features` reads every episode's joint and gripper states, which all episodes record, so the features do not depend on the action space. Speeds and idle times are relative to the episode's own 95th percentile joint speed, which makes episodes of different robots comparable.

| Feature | Meaning |
|---|---|
| `mean_speed` | mean joint speed (rad/s) |
| `idle_share` | share of steps below `idle_speed_fraction` of the 95th percentile speed |
| `path_efficiency` | joint-space distance from start to end over the distance travelled |
| `log_jerk` | log dimensionless jerk of the joint speed profile; higher is smoother |
| `oscillation` | joint velocity sign changes per joint and second, ignoring idle motion |
| `sudden_stops` | falls from above half the 95th percentile speed to idle within `sudden_stop_seconds`, per minute |
| `gripper_events` | changes between the `gripper_low` and `gripper_high` levels of the robot's normalized gripper range, per minute |
| `regrasp` | two or more gripper closes |
| `closed_on_nothing` | a close that reaches `gripper_full`, i.e. closed on nothing |
| `tracking_gap` | distance between a joint position command and the next joint state over the mean step; only for episodes that command joint positions |

A robot's gripper range spans the 5th percentile of its episodes' gripper minima to the 95th percentile of their maxima, so extraction reads every file twice. `regrasp` and `closed_on_nothing` need to know which end of the range is closed; they are computed only for robots listed in `gripper_closed`. Episodes without a joint state are skipped.

With `--segments`, `features` also computes the features in a window of `window_seconds` before and after every segment start. Entry 38 reports them as differences to the features of the whole episode.

Entry 41 trains a logistic regression on the features to tell failed from successful episodes. Its AUROC is cross-validated with whole labs held out, so a model cannot score by recognizing a lab.
