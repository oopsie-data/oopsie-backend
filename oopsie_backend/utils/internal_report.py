"""Render a self-contained HTML internal report from internal_stats.json."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

_LANG_STACK_COLORS = (
    "#0ea5e9",
    "#8b5cf6",
    "#f59e0b",
    "#10b981",
    "#ef4444",
    "#ec4899",
    "#6366f1",
    "#14b8a6",
    "#f97316",
    "#84cc16",
)


def _repo_short_name(full_id: str) -> str:
    return full_id.split("/", 1)[-1] if "/" in full_id else full_id


def _dataset_bar_rows(per_dataset: dict[str, dict]) -> str:
    if not per_dataset:
        return '<p class="empty">No data</p>'
    items = sorted(
        per_dataset.items(),
        key=lambda kv: (-(kv[1].get("episodes") or 0), kv[0]),
    )
    top = items[0][1].get("episodes") or 1
    rows = []
    for full_id, stats in items:
        count = int(stats.get("episodes") or 0)
        hours = stats.get("duration_hours")
        bulk = bool(stats.get("bulk_import"))
        width = max(4, round(100 * count / top))
        fill_class = "bar-fill bar-fill-bulk" if bulk else "bar-fill"
        short = _repo_short_name(full_id)
        tip = full_id
        if hours is not None:
            tip += f" · {hours:.2f} h"
        if bulk:
            tip += " · bulk import (excluded from public stats)"
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{html.escape(tip)}">{html.escape(short)}</span>'
            f'<div class="bar-track"><div class="{fill_class}" style="width:{width}%"></div></div>'
            f'<span class="bar-count">{count:,}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _dataset_section(per_dataset: dict[str, dict], totals: dict) -> str:
    n_repos = totals.get("all_datasets") or len(per_dataset)
    n_eps = totals.get("all_dataset_episodes") or sum(
        int(v.get("episodes") or 0) for v in per_dataset.values()
    )
    bulk_count = sum(1 for v in per_dataset.values() if v.get("bulk_import"))
    return f"""
<section class="dataset-section">
  <div class="section-head">
    <h2>Episodes by Hugging Face repo</h2>
    <p class="section-desc">{n_repos} repos · {n_eps:,} episodes total · {bulk_count} bulk-import repo(s) marked in amber</p>
  </div>
  <div class="panel dataset-panel">
    {_dataset_bar_rows(per_dataset)}
  </div>
</section>
"""


def _bar_rows(counts: dict[str, int], max_items: int = 30) -> str:
    if not counts:
        return '<p class="empty">No data</p>'
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_items]
    top = items[0][1] if items else 1
    rows = []
    for name, count in items:
        width = max(4, round(100 * count / top))
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{html.escape(name)}">{html.escape(name)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>'
            f'<span class="bar-count">{count:,}</span>'
            f"</div>"
        )
    if len(counts) > max_items:
        rest = len(counts) - max_items
        rows.append(f'<p class="truncated">+ {rest} more not shown</p>')
    return "\n".join(rows)


def _lab_section(lab_id: str, lab: dict) -> str:
    operators = lab.get("operators") or {}
    annotators = lab.get("annotators") or {}
    op_eps = lab.get("operator_episodes", sum(operators.values()))
    ann_count = lab.get("annotation_count", sum(annotators.values()))
    return f"""
<section class="lab-fold" data-lab="{html.escape(lab_id.lower())}">
  <button type="button" class="lab-toggle" aria-expanded="false">
    <span class="lab-name">{html.escape(lab_id)}</span>
    <span class="lab-meta">{op_eps:,} episodes · {ann_count:,} annotations · {len(operators)} operators · {len(annotators)} annotators</span>
    <span class="chevron" aria-hidden="true"></span>
  </button>
  <div class="lab-body" hidden>
    <div class="lab-grid">
      <div class="panel">
        <h3>Operators <span class="hint">(episodes collected)</span></h3>
        {_bar_rows(operators)}
      </div>
      <div class="panel">
        <h3>Annotators <span class="hint">(complete annotations)</span></h3>
        {_bar_rows(annotators)}
      </div>
    </div>
  </div>
</section>
"""


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _language_top_rows(top_strings: list[dict]) -> str:
    if not top_strings:
        return '<p class="empty">No data</p>'
    top = top_strings[0].get("episodes") or 1
    rows = []
    for item in top_strings:
        text = str(item.get("text") or "")
        count = int(item.get("episodes") or 0)
        share = float(item.get("share") or 0.0)
        width = max(4, round(100 * count / top))
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{html.escape(text)}">{html.escape(text)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>'
            f'<span class="bar-count">{count:,} · {_pct(share)}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _language_per_lab_rows(per_lab: dict[str, dict]) -> str:
    if not per_lab:
        return '<p class="empty">No data</p>'
    rows = [
        "<table class=\"lang-table lang-lab-table\">",
        "<thead><tr>"
        "<th>Lab</th><th>Episodes</th><th>Unique</th><th>Norm. unique</th>"
        "<th>Eff. vocab</th><th>Top 5 share</th>"
        "</tr></thead><tbody>",
    ]
    for lab, stats in sorted(per_lab.items()):
        rows.append(
            f'<tr class="lang-lab-row" data-lab="{html.escape(lab, quote=True)}" tabindex="0" role="button" '
            f'title="Show unique strings for {html.escape(lab)}">'
            f"<td>{html.escape(lab)}</td>"
            f"<td>{int(stats.get('episodes') or 0):,}</td>"
            f"<td>{int(stats.get('raw_unique') or 0):,}</td>"
            f"<td>{int(stats.get('normalized_unique') or 0):,}</td>"
            f"<td>{stats.get('effective_vocabulary', 0)}</td>"
            f"<td>{_pct(float(stats.get('top_5_share') or 0.0))}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    rows.append('<p class="lang-table-hint">Click a lab row to view its unique strings.</p>')
    return "\n".join(rows)


def _language_per_lab_stacked_bars(per_lab_strings: dict[str, list[dict]]) -> str:
    """Horizontal stacked bars: once-only strings vs each repeated exact string."""
    if not per_lab_strings:
        return '<p class="empty">No data</p>'

    lab_rows: list[tuple[str, int, list[dict]]] = []
    for lab, strings in per_lab_strings.items():
        total = sum(int(item.get("episodes") or 0) for item in strings)
        if total > 0:
            lab_rows.append((lab, total, strings))
    lab_rows.sort(key=lambda row: (-row[1], row[0]))

    rows = ['<div class="lang-stack-chart">']
    for lab, total, strings in lab_rows:
        once_only = [item for item in strings if int(item.get("episodes") or 0) == 1]
        repeated = sorted(
            (item for item in strings if int(item.get("episodes") or 0) > 1),
            key=lambda item: (-int(item.get("episodes") or 0), str(item.get("text") or "")),
        )
        singleton_eps = len(once_only)
        segments: list[str] = []
        if singleton_eps:
            distinct_once = len(once_only)
            tip = html.escape(
                f"Once-only strings: {singleton_eps:,} episodes "
                f"({distinct_once:,} distinct exact strings, each used once)",
                quote=True,
            )
            segments.append(
                f'<span class="lang-stack-seg lang-stack-singleton" style="flex:{singleton_eps} 0 0" '
                f'title="{tip}"></span>'
            )
        for index, item in enumerate(repeated):
            episodes = int(item.get("episodes") or 0)
            text = str(item.get("text") or "")
            color = _LANG_STACK_COLORS[index % len(_LANG_STACK_COLORS)]
            tip = html.escape(f"{text}: {episodes:,} episodes", quote=True)
            segments.append(
                f'<span class="lang-stack-seg" style="flex:{episodes} 0 0;background:{color}" '
                f'title="{tip}"></span>'
            )
        rows.append(
            f'<div class="lang-stack-row">'
            f'<span class="lang-stack-label" title="{html.escape(lab, quote=True)}">{html.escape(lab)}</span>'
            f'<div class="lang-stack-track">{"".join(segments)}</div>'
            f'<span class="lang-stack-total">{total:,}</span>'
            f"</div>"
        )
    rows.append("</div>")
    rows.append(
        '<p class="lang-stack-legend">'
        '<span class="lang-stack-swatch lang-stack-singleton"></span> once-only exact strings · '
        "colored segments = strings repeated 2+ times (hover for wording and count)"
        "</p>"
    )
    return "\n".join(rows)


def _length_axis_pct(value: float, scale_max: float) -> float:
    if scale_max <= 0:
        return 0.0
    return min(100.0, max(0.0, 100.0 * value / scale_max))


def _failure_desc_length_boxplot_chart(length_dist: dict) -> str:
    per_lab = length_dist.get("per_lab") or {}
    if not per_lab:
        return '<p class="empty">No described failures.</p>'

    corpus_box = length_dist.get("corpus_box") or {}
    corpus_median = float(corpus_box.get("median_words") or length_dist.get("corpus_median_words") or 0)
    rows_data = sorted(
        (
            (lab, stats)
            for lab, stats in per_lab.items()
            if int(stats.get("count") or 0) > 0
        ),
        key=lambda item: (-float(item[1].get("median_words") or 0), item[0]),
    )
    scale_max = max(
        (
            float(stats.get("max_words") or stats.get("whisker_high_words") or 0)
            for _, stats in rows_data
        ),
        default=1.0,
    )
    scale_max = max(scale_max, corpus_median, 1.0)
    scale_max = math.ceil(scale_max * 1.05)

    ref_pct = _length_axis_pct(corpus_median, scale_max) if corpus_median > 0 else 0.0
    ref_marker = (
        f'<div class="lang-len-box-ref" style="left:{ref_pct:.2f}%" '
        f'title="Corpus median: {corpus_median:g} words"></div>'
        if corpus_median > 0
        else ""
    )

    rows = ['<div class="lang-len-boxplot-chart">']
    for lab, stats in rows_data:
        count = int(stats.get("count") or 0)
        p25 = float(stats.get("p25_words") or 0)
        median = float(stats.get("median_words") or 0)
        p75 = float(stats.get("p75_words") or 0)
        whisker_low = float(stats.get("whisker_low_words") or stats.get("min_words") or 0)
        whisker_high = float(stats.get("whisker_high_words") or stats.get("max_words") or 0)
        min_words = float(stats.get("min_words") or whisker_low)
        max_words = float(stats.get("max_words") or whisker_high)
        outlier_count = int(stats.get("outlier_count") or 0)

        w_left = _length_axis_pct(whisker_low, scale_max)
        w_width = _length_axis_pct(whisker_high, scale_max) - w_left
        box_left = _length_axis_pct(p25, scale_max)
        box_width = max(0.4, _length_axis_pct(p75, scale_max) - box_left)
        med_left = _length_axis_pct(median, scale_max)

        tip = (
            f"{lab}: Q1={p25:g}, median={median:g}, Q3={p75:g}, "
            f"whiskers {whisker_low:g}–{whisker_high:g} words (n={count:,})"
        )
        if outlier_count:
            tip += f", {outlier_count} outlier(s) beyond whiskers"

        outlier_dots: list[str] = []
        if min_words < whisker_low - 1e-9:
            outlier_dots.append(
                f'<span class="lang-len-box-outlier" style="left:{_length_axis_pct(min_words, scale_max):.2f}%" '
                f'title="Low outlier: {min_words:g} words"></span>'
            )
        if max_words > whisker_high + 1e-9:
            outlier_dots.append(
                f'<span class="lang-len-box-outlier" style="left:{_length_axis_pct(max_words, scale_max):.2f}%" '
                f'title="High outlier: {max_words:g} words"></span>'
            )

        rows.append(
            f'<div class="lang-len-box-row" title="{html.escape(tip, quote=True)}">'
            f'<span class="lang-stack-label" title="{html.escape(lab, quote=True)}">{html.escape(lab)}</span>'
            f'<div class="lang-len-box-track-wrap">'
            f"{ref_marker}"
            f'<div class="lang-len-box-track">'
            f'<span class="lang-len-box-whisker" style="left:{w_left:.2f}%;width:{w_width:.2f}%"></span>'
            f'<span class="lang-len-box-cap is-low" style="left:{w_left:.2f}%"></span>'
            f'<span class="lang-len-box-cap is-high" style="left:{_length_axis_pct(whisker_high, scale_max):.2f}%"></span>'
            f'<span class="lang-len-box-rect" style="left:{box_left:.2f}%;width:{box_width:.2f}%"></span>'
            f'<span class="lang-len-box-median" style="left:{med_left:.2f}%"></span>'
            f"{''.join(outlier_dots)}"
            f"</div></div>"
            f'<span class="lang-len-median-val">{median:g} · n={count:,}</span>'
            f"</div>"
        )

    tick_marks = [0, scale_max // 2, scale_max] if scale_max >= 4 else [0, scale_max]
    tick_marks = sorted(set(int(t) for t in tick_marks))
    axis = "".join(
        f'<span class="lang-len-box-tick" style="left:{_length_axis_pct(t, scale_max):.2f}%">'
        f"{t}</span>"
        for t in tick_marks
    )
    rows.append(
        f'<div class="lang-len-box-axis-row">'
        f'<span class="lang-stack-label lang-len-box-axis-label">Words</span>'
        f'<div class="lang-len-box-axis">{axis}</div>'
        f'<span class="lang-len-median-val"></span>'
        f"</div>"
    )
    rows.append("</div>")

    c_q1 = float(corpus_box.get("p25_words") or 0)
    c_q3 = float(corpus_box.get("p75_words") or 0)
    c_wh = float(corpus_box.get("whisker_high_words") or 0)
    c_wl = float(corpus_box.get("whisker_low_words") or 0)
    rows.append(
        f'<p class="lang-len-note">Box = Q1–Q3, line = median, whiskers = 1.5×IQR fences. '
        f"Corpus: median <strong>{corpus_median:g}</strong>, IQR {c_q1:g}–{c_q3:g}, "
        f"whiskers {c_wl:g}–{c_wh:g}. Red marker = corpus median."
    )
    return "\n".join(rows)


def _failure_desc_length_median_chart(length_dist: dict) -> str:
    per_lab = length_dist.get("per_lab") or {}
    if not per_lab:
        return '<p class="empty">No described failures.</p>'

    corpus_median = float(length_dist.get("corpus_median_words") or 0)
    corpus_count = int(length_dist.get("corpus_count") or 0)
    rows_data = sorted(
        (
            (lab, stats)
            for lab, stats in per_lab.items()
            if int(stats.get("count") or 0) > 0
        ),
        key=lambda item: (-float(item[1].get("median_words") or 0), item[0]),
    )
    max_median = max(float(stats.get("median_words") or 0) for _, stats in rows_data) or 1.0
    ref_pct = min(100, round(100 * corpus_median / max_median)) if corpus_median > 0 else 0
    ref_marker = (
        f'<div class="lang-len-median-ref" style="left:{ref_pct}%" '
        f'title="Corpus median: {corpus_median:g} words"></div>'
        if corpus_median > 0
        else ""
    )

    rows = ['<div class="lang-len-median-chart">']
    for lab, stats in rows_data:
        median = float(stats.get("median_words") or 0)
        count = int(stats.get("count") or 0)
        width = max(2, round(100 * median / max_median))
        tip = f"Median {median:g} words (n={count:,})"
        rows.append(
            f'<div class="lang-len-median-row">'
            f'<span class="lang-stack-label" title="{html.escape(lab, quote=True)}">{html.escape(lab)}</span>'
            f'<div class="lang-len-median-track-wrap">'
            f"{ref_marker}"
            f'<div class="lang-len-median-track">'
            f'<div class="lang-len-median-fill" style="width:{width}%" '
            f'title="{html.escape(tip, quote=True)}"></div>'
            f"</div></div>"
            f'<span class="lang-len-median-val">{median:g} · n={count:,}</span>'
            f"</div>"
        )
    rows.append("</div>")
    rows.append(
        f'<p class="lang-len-note">Corpus median: <strong>{corpus_median:g}</strong> words '
        f"among {corpus_count:,} described failures. "
        "Red marker = corpus median relative to the longest lab median bar."
    )
    return "\n".join(rows)


def _failure_desc_length_bucket_block(length_dist: dict) -> str:
    thresholds = length_dist.get("bucket_thresholds") or {}
    short_max = int(thresholds.get("short_max_words") or 5)
    medium_max = int(thresholds.get("medium_max_words") or 15)
    payload = json.dumps(length_dist.get("per_lab") or {}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    return f"""
<div class="lang-len-buckets-wrap">
  <div class="lang-len-bucket-toolbar">
    <p id="lang-len-bucket-desc" class="lang-stack-desc">
      Each bar spans the full width so you can compare short / medium / long shares across labs.
    </p>
    <div id="lang-len-bucket-seg" class="lang-len-seg" role="group" aria-label="Length bucket view">
      <button type="button" class="lang-len-seg-opt" data-mode="rate" aria-selected="true">Compare rates</button>
      <button type="button" class="lang-len-seg-opt" data-mode="volume">By volume</button>
    </div>
  </div>
  <div id="lang-len-bucket-chart" class="lang-len-bucket-chart"></div>
  <script type="application/json" id="lang-fail-length-data">{payload}</script>
</div>
<p class="lang-stack-legend">
  <span class="lang-len-swatch lang-len-swatch-short"></span> short (≤{short_max} words) ·
  <span class="lang-len-swatch lang-len-swatch-medium"></span> medium ({short_max + 1}–{medium_max}) ·
  <span class="lang-len-swatch lang-len-swatch-long"></span> long ({medium_max + 1}+)
  · among described failures only
</p>
"""


def _nice_axis_max(value: int) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(value)))
    return int(math.ceil(value / magnitude) * magnitude)


def _instruction_verb_chart(verb_distribution: dict) -> str:
    verbs = verb_distribution.get("verbs") or []
    if not verbs:
        return ""

    max_count = max(int(item.get("episodes") or 0) for item in verbs) or 1
    axis_max = _nice_axis_max(max_count)
    mid_tick = axis_max // 2
    total_eps = int(verb_distribution.get("total_episodes") or 0)
    unique_verbs = int(verb_distribution.get("unique_verbs") or 0)

    bars: list[str] = []
    for index, item in enumerate(verbs):
        verb = str(item.get("verb") or "")
        count = int(item.get("episodes") or 0)
        share = float(item.get("share") or 0.0)
        height = max(1.5, round(1000 * count / axis_max) / 10)
        color = _LANG_STACK_COLORS[index % len(_LANG_STACK_COLORS)]
        tip = f"{verb}: {count:,} episodes ({_pct(share)})"
        bars.append(
            f'<div class="lang-verb-col" title="{html.escape(tip, quote=True)}">'
            f'<div class="lang-verb-bar-wrap">'
            f'<div class="lang-verb-bar" style="height:{height}%;background:{color}"></div>'
            f"</div>"
            f'<div class="lang-verb-label">{html.escape(verb)}</div>'
            f"</div>"
        )

    return f"""
<div class="lang-verb-section">
  <h3>Task instruction verbs</h3>
  <p class="lang-stack-desc">
    Episode-weighted distribution of the primary verb in each task instruction
    (cf. <a href="https://droid-dataset.github.io/">DROID skill analysis</a>).
    Heuristic: phrasal verbs (e.g. pick up → pick), skip light verbs like perform,
    then first content word lemmatized. {unique_verbs:,} distinct verbs across
    {total_eps:,} episodes with instructions; chart shows top {len(verbs)} classes.
  </p>
  <div class="lang-verb-chart">
    <div class="lang-verb-yaxis" aria-hidden="true">
      <span>{axis_max:,}</span>
      <span>{mid_tick:,}</span>
      <span>0</span>
    </div>
    <div class="lang-verb-plot">
      <div class="lang-verb-bars">
        {"".join(bars)}
      </div>
    </div>
  </div>
</div>
"""


def _failure_desc_length_section(failures: dict) -> str:
    length_dist = failures.get("length_distribution") or {}
    per_lab = length_dist.get("per_lab") or {}
    if not per_lab:
        return ""
    return f"""
<div class="lang-len-section">
  <h3>Failure description length (per lab)</h3>
  <p class="lang-stack-desc">
    Word counts for annotated failures with non-empty text only — selection bias vs the coverage chart above.
  </p>
  <h4>Median words</h4>
  {_failure_desc_length_median_chart(length_dist)}
  <h4>Word-count distribution</h4>
  {_failure_desc_length_boxplot_chart(length_dist)}
  <h4>Length mix</h4>
  {_failure_desc_length_bucket_block(length_dist)}
</div>
"""


def _language_lab_drilldown_payload(language: dict) -> dict[str, dict[str, list[dict]]]:
    tasks = language.get("task_instructions") or {}
    failures = language.get("failure_descriptions") or {}
    task_strings = tasks.get("per_lab_strings") or {}
    fail_strings = failures.get("per_lab_strings") or {}
    labs = sorted(set(task_strings) | set(fail_strings))
    return {
        lab: {
            "instructions": task_strings.get(lab, []),
            "failures": fail_strings.get(lab, []),
        }
        for lab in labs
    }


def _language_corpus_drilldown_payload(language: dict) -> dict[str, object]:
    tasks = language.get("task_instructions") or {}
    return {
        "instructions_normalized": tasks.get("normalized_strings") or [],
        "instructions_near_duplicate_groups": tasks.get("near_duplicate_group_list") or [],
        "near_duplicate_threshold": tasks.get("near_duplicate_threshold", 0.9),
        "normalized_unique": int(tasks.get("normalized_unique") or 0),
        "near_duplicate_merges": int(tasks.get("near_duplicate_merges") or 0),
    }


def _language_diversity_section(language: dict | None) -> str:
    if not language:
        return ""
    tasks = language.get("task_instructions") or {}
    failures = language.get("failure_descriptions") or {}
    excluded = language.get("excluded_repos") or []
    excluded_note = ", ".join(_repo_short_name(r) for r in excluded[:4])
    if len(excluded) > 4:
        excluded_note += f", +{len(excluded) - 4} more"
    lab_payload = json.dumps(_language_lab_drilldown_payload(language), ensure_ascii=False)
    lab_payload = lab_payload.replace("</", "<\\/")
    corpus_payload = json.dumps(_language_corpus_drilldown_payload(language), ensure_ascii=False)
    corpus_payload = corpus_payload.replace("</", "<\\/")
    return f"""
<section class="dataset-section language-section">
  <div class="section-head">
    <h2>Language diversity (public corpus)</h2>
    <p class="section-desc">
      Tier 1 lexical metrics on episodes included on the public website.
      Bulk-import repos excluded ({html.escape(excluded_note)}).
      Failure-description metrics cover annotated failures with non-empty text only
      ({int(failures.get('with_description') or 0):,} of {int(failures.get('total_failures') or 0):,} failures,
      {_pct(float(failures.get('description_coverage') or 0.0))} coverage).
    </p>
  </div>
  <div class="lang-grid">
    {_language_corpus_panel("Task instructions", tasks, corpus_id="instructions")}
    {_language_corpus_panel("Failure descriptions", failures, corpus_id="failures")}
  </div>
  {_instruction_verb_chart(tasks.get("verb_distribution") or {})}
  {_failure_desc_length_section(failures)}
  <div id="lang-lab-detail" class="lang-lab-detail" hidden>
    <div class="lang-lab-detail-head">
      <h3>Lab strings: <span id="lang-lab-detail-name"></span></h3>
      <button type="button" id="lang-lab-detail-close" class="lang-detail-close">Close</button>
    </div>
    <div class="lang-detail-grid">
      <div class="lang-detail-panel">
        <h4>Task instructions (<span id="lang-inst-count">0</span> unique)</h4>
        <div class="lang-detail-scroll">
          <table class="lang-table lang-string-table">
            <thead><tr><th>Instruction</th><th>Episodes</th><th>Share</th></tr></thead>
            <tbody id="lang-inst-rows"></tbody>
          </table>
        </div>
      </div>
      <div class="lang-detail-panel">
        <h4>Failure descriptions (<span id="lang-fail-count">0</span> unique)</h4>
        <div class="lang-detail-scroll">
          <table class="lang-table lang-string-table">
            <thead><tr><th>Description</th><th>Episodes</th><th>Share</th></tr></thead>
            <tbody id="lang-fail-rows"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
  <div id="lang-corpus-detail" class="lang-lab-detail" hidden>
    <div class="lang-lab-detail-head">
      <h3><span id="lang-corpus-detail-title">All unique strings</span> (<span id="lang-corpus-detail-count">0</span>)</h3>
      <button type="button" id="lang-corpus-detail-close" class="lang-detail-close">Close</button>
    </div>
    <p id="lang-corpus-detail-desc" class="section-desc"></p>
    <div class="lang-detail-scroll lang-detail-scroll-wide">
      <table class="lang-table lang-string-table">
        <thead id="lang-corpus-head"><tr><th>String</th><th>Episodes</th><th>Share</th></tr></thead>
        <tbody id="lang-corpus-rows"></tbody>
      </table>
    </div>
  </div>
  <script type="application/json" id="lang-lab-data">{lab_payload}</script>
  <script type="application/json" id="lang-corpus-data">{corpus_payload}</script>
</section>
"""


def _language_corpus_panel(title: str, corpus: dict, *, corpus_id: str | None = None) -> str:
    concentration = corpus.get("concentration") or {}
    eps_per = corpus.get("episodes_per_string") or {}
    norm_unique = int(corpus.get("normalized_unique") or 0)
    near_dup = int(corpus.get("near_duplicate_groups") or 0)
    if corpus_id == "instructions":
        norm_stat = (
            f'<div class="stat lang-stat-clickable lang-corpus-stat" '
            f'data-view="normalized" data-corpus="{html.escape(corpus_id, quote=True)}" '
            f'tabindex="0" role="button" '
            f'title="Show all unique task instructions (normalized)">'
            f'<div class="val">{norm_unique:,}</div><div class="lbl">Normalized unique</div></div>'
        )
        near_stat = (
            f'<div class="stat lang-stat-clickable lang-corpus-stat" '
            f'data-view="near_duplicate" data-corpus="{html.escape(corpus_id, quote=True)}" '
            f'tabindex="0" role="button" '
            f'title="Show near-duplicate groups (≥90% character similarity)">'
            f'<div class="val">{near_dup:,}</div><div class="lbl">Near-dup groups</div></div>'
        )
    else:
        norm_stat = (
            f'<div class="stat"><div class="val">{norm_unique:,}</div>'
            f'<div class="lbl">Normalized unique</div></div>'
        )
        near_stat = (
            f'<div class="stat"><div class="val">{near_dup:,}</div><div class="lbl">Near-dup groups</div></div>'
        )
    return f"""
<div class="lang-panel">
  <h3>{html.escape(title)}</h3>
  <div class="lang-metrics">
    <div class="stat"><div class="val">{int(corpus.get('total_episodes') or 0):,}</div><div class="lbl">Episodes</div></div>
    <div class="stat"><div class="val">{int(corpus.get('raw_unique') or 0):,}</div><div class="lbl">Exact unique</div></div>
    {norm_stat}
    {near_stat}
    <div class="stat"><div class="val">{corpus.get('effective_vocabulary', 0)}</div><div class="lbl">Effective vocab</div></div>
    <div class="stat"><div class="val">{_pct(float(concentration.get('top_5_share') or 0.0))}</div><div class="lbl">Top 5 share</div></div>
  </div>
  <p class="section-desc">
    Entropy {corpus.get('entropy_bits', 0)} bits ·
    top 10 {_pct(float(concentration.get('top_10_share') or 0.0))} ·
    top 20 {_pct(float(concentration.get('top_20_share') or 0.0))} ·
    median {eps_per.get('median', 0):.0f} eps/string (p90 {eps_per.get('p90', 0):.0f}, max {int(eps_per.get('max') or 0):,})
  </p>
  <h4>Most frequent strings</h4>
  {_language_top_rows(corpus.get("top_strings") or [])}
  <h4>Per lab <span class="hint">(click a row)</span></h4>
  {_language_per_lab_rows(corpus.get("per_lab") or {})}
  <h4>Per lab: repetition mix</h4>
  <p class="lang-stack-desc">Each bar is all episodes for that lab. Gray = strings used exactly once; each colored block = one exact string used multiple times.</p>
  {_language_per_lab_stacked_bars(corpus.get("per_lab_strings") or {})}
</div>
"""


def _normalize_internal_data(data: dict) -> dict:
    """Support legacy flat JSON and schema v3 nested internal reports."""
    if "stats" not in data:
        return data
    stats = data.get("stats") or {}
    people = data.get("people") or {}
    st = stats.get("totals") or {}
    pt = people.get("totals") or {}
    return {
        "generated_at": data.get("generated_at") or stats.get("generated_at") or {},
        "dataset_repo": data.get("dataset_repo") or stats.get("dataset_repo") or "",
        "excluded_from_public_datasets": data.get("excluded_from_public_datasets") or [],
        "stats_totals": st,
        "per_dataset": stats.get("per_dataset") or {},
        "per_lab": people.get("per_lab") or {},
        "language_diversity": data.get("language_diversity"),
        "totals": {
            **st,
            "operator_episodes": pt.get("operator_episodes", 0),
            "annotations": pt.get("annotations", 0),
            "all_dataset_episodes": st.get("episodes", 0),
            "all_datasets": st.get("datasets", 0),
        },
    }


def _stats_summary_section(stats_totals: dict) -> str:
    if not stats_totals:
        return ""
    t = stats_totals
    return f"""
<section class="dataset-section stats-summary">
  <div class="section-head">
    <h2>Full dataset totals</h2>
    <p class="section-desc">Includes bulk-import repos excluded from the public website.</p>
  </div>
  <div class="summary">
    <div class="stat"><div class="val">{t.get('episodes', 0):,}</div><div class="lbl">Episodes</div></div>
    <div class="stat"><div class="val">{t.get('labs', 0):,}</div><div class="lbl">Labs</div></div>
    <div class="stat"><div class="val">{t.get('datasets', 0):,}</div><div class="lbl">HF repos</div></div>
    <div class="stat"><div class="val">{t.get('successes', 0):,}</div><div class="lbl">Successes</div></div>
    <div class="stat"><div class="val">{t.get('failures', 0):,}</div><div class="lbl">Failures</div></div>
    <div class="stat"><div class="val">{t.get('duration_hours', 0):,.1f}</div><div class="lbl">Hours</div></div>
  </div>
</section>
"""


def render_internal_report_html(data: dict) -> str:
    data = _normalize_internal_data(data)
    generated = data.get("generated_at") or {}
    when = generated.get("cdt") or generated.get("utc") or "unknown"
    totals = data.get("totals") or {}
    per_lab = data.get("per_lab") or {}
    per_dataset = data.get("per_dataset") or {}
    stats_totals = data.get("stats_totals") or {}
    language_diversity = data.get("language_diversity")
    labs_html = "\n".join(_lab_section(lab, per_lab[lab]) for lab in sorted(per_lab))
    dataset_html = _dataset_section(per_dataset, totals) if per_dataset else ""
    stats_summary_html = _stats_summary_section(stats_totals)
    language_html = _language_diversity_section(language_diversity)
    excluded = data.get("excluded_from_public_datasets") or []
    excluded_note = ", ".join(_repo_short_name(r) for r in excluded[:4])
    if len(excluded) > 4:
        excluded_note += f", +{len(excluded) - 4} more"
    people_scope_note = (
        "Public corpus only — bulk-import repos excluded"
        + (f" ({html.escape(excluded_note)})" if excluded_note else "")
        + ". Annotations count only when success is set; failures also require "
        "description, category, and severity."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OopsieData — internal stats</title>
  <style>
    :root {{
      --surface: #ffffff;
      --surface-2: #f8fafc;
      --border: #e2e8f0;
      --ink: #0f172a;
      --ink-muted: #64748b;
      --accent: #0ea5e9;
      --radius: 12px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: var(--ink);
      background: var(--surface-2);
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }}
    header {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
    }}
    header h1 {{ margin: 0 0 0.35rem; font-size: 1.35rem; }}
    header p {{ margin: 0.15rem 0; color: var(--ink-muted); font-size: 0.92rem; }}
    .badge {{
      display: inline-block;
      margin-top: 0.65rem;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: #fef3c7;
      color: #92400e;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.65rem;
      margin-bottom: 1rem;
    }}
    .stat {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
    }}
    .stat .val {{ font-size: 1.45rem; font-weight: 700; font-feature-settings: "tnum"; }}
    .stat .lbl {{ font-size: 0.78rem; color: var(--ink-muted); margin-top: 0.15rem; }}
    .dataset-section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.1rem 1.25rem;
      margin-bottom: 1rem;
    }}
    .section-head h2 {{
      margin: 0 0 0.25rem;
      font-size: 1.05rem;
    }}
    .section-desc {{
      margin: 0 0 0.85rem;
      font-size: 0.85rem;
      color: var(--ink-muted);
    }}
    .dataset-panel {{ margin: 0; }}
    .bar-fill-bulk {{
      background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }}
    .labs-section-head {{
      margin: 0 0 0.75rem;
      font-size: 1.05rem;
    }}
    .toolbar {{
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 0.75rem;
    }}
    .toolbar input {{
      flex: 1;
      padding: 0.55rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    .toolbar button {{
      padding: 0.55rem 0.85rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      cursor: pointer;
      font-size: 0.85rem;
    }}
    .lab-fold {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      margin-bottom: 0.55rem;
      overflow: hidden;
    }}
    .lab-toggle {{
      width: 100%;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.85rem 1rem;
      border: 0;
      background: transparent;
      cursor: pointer;
      text-align: left;
      font: inherit;
      color: inherit;
    }}
    .lab-toggle:hover {{ background: #f1f5f9; }}
    .lab-name {{ font-weight: 600; flex: 0 0 auto; }}
    .lab-meta {{ flex: 1; color: var(--ink-muted); font-size: 0.85rem; }}
    .chevron {{
      width: 0.55rem; height: 0.55rem;
      border-right: 2px solid var(--ink-muted);
      border-bottom: 2px solid var(--ink-muted);
      transform: rotate(-45deg);
      transition: transform 160ms ease;
    }}
    .lab-fold.is-open .chevron {{ transform: rotate(45deg); }}
    .lab-body {{ padding: 0 1rem 1rem; }}
    .lab-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 760px) {{ .lab-grid {{ grid-template-columns: 1fr; }} }}
    .panel h3 {{
      margin: 0 0 0.65rem;
      font-size: 0.95rem;
    }}
    .panel .hint {{ color: var(--ink-muted); font-weight: 400; font-size: 0.8rem; }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(80px, 28%) 1fr auto;
      gap: 0.5rem;
      align-items: center;
      margin-bottom: 0.35rem;
      font-size: 0.85rem;
    }}
    .bar-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-track {{
      height: 0.55rem;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #38bdf8);
      border-radius: 999px;
      min-width: 2px;
    }}
    .bar-count {{
      font-feature-settings: "tnum";
      color: var(--ink-muted);
      min-width: 3rem;
      text-align: right;
    }}
    .empty, .truncated {{ color: var(--ink-muted); font-size: 0.85rem; margin: 0; }}
    .no-match {{
      display: none;
      padding: 1rem;
      text-align: center;
      color: var(--ink-muted);
    }}
    .language-section .lang-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 960px) {{ .language-section .lang-grid {{ grid-template-columns: 1fr; }} }}
    .lang-panel {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      background: var(--surface-2);
    }}
    .lang-panel h3 {{
      margin: 0 0 0.65rem;
      font-size: 0.98rem;
    }}
    .lang-panel h4 {{
      margin: 0.85rem 0 0.45rem;
      font-size: 0.85rem;
      color: var(--ink-muted);
      font-weight: 600;
    }}
    .lang-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 0.45rem;
      margin-bottom: 0.35rem;
    }}
    .lang-metrics .stat {{
      padding: 0.55rem 0.65rem;
    }}
    .lang-metrics .stat .val {{
      font-size: 1.05rem;
    }}
    .lang-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }}
    .lang-table th, .lang-table td {{
      border-bottom: 1px solid var(--border);
      padding: 0.35rem 0.45rem;
      text-align: left;
    }}
    .lang-table th {{
      color: var(--ink-muted);
      font-weight: 600;
    }}
    .lang-table-hint {{
      margin: 0.35rem 0 0;
      font-size: 0.75rem;
      color: var(--ink-muted);
    }}
    .lang-lab-row {{
      cursor: pointer;
    }}
    .lang-lab-row:hover td {{
      background: #f1f5f9;
    }}
    .lang-lab-row.is-selected td {{
      background: #e0f2fe;
    }}
    .lang-lab-row td:first-child {{
      color: var(--accent);
      font-weight: 600;
    }}
    .lang-lab-detail {{
      margin-top: 1rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      background: var(--surface);
    }}
    .lang-lab-detail-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.65rem;
    }}
    .lang-lab-detail-head h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    .lang-detail-close {{
      padding: 0.35rem 0.65rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface-2);
      cursor: pointer;
      font-size: 0.82rem;
    }}
    .lang-detail-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 960px) {{ .lang-detail-grid {{ grid-template-columns: 1fr; }} }}
    .lang-detail-panel h4 {{
      margin: 0 0 0.45rem;
      font-size: 0.88rem;
    }}
    .lang-detail-scroll {{
      max-height: 360px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .lang-string-table td:first-child {{
      white-space: normal;
      word-break: break-word;
      min-width: 180px;
    }}
    .lang-string-table td:not(:first-child) {{
      white-space: nowrap;
      text-align: right;
      font-feature-settings: "tnum";
    }}
    .lang-string-table .empty-row td {{
      color: var(--ink-muted);
      font-style: italic;
      text-align: left;
    }}
    .lang-stat-clickable {{
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease;
    }}
    .lang-stat-clickable:hover {{
      background: #f0f9ff;
      border-color: #7dd3fc;
    }}
    .lang-stat-clickable.is-selected {{
      background: #e0f2fe;
      border-color: #38bdf8;
    }}
    .lang-stat-clickable .lbl {{
      text-decoration: underline dotted;
      text-underline-offset: 2px;
    }}
    .lang-detail-scroll-wide {{
      max-height: 480px;
    }}
    .lang-variant-note {{
      display: block;
      margin-top: 0.15rem;
      font-size: 0.72rem;
      color: var(--ink-muted);
    }}
    .lang-member-list {{
      margin: 0;
      padding-left: 1rem;
      font-size: 0.78rem;
    }}
    .lang-member-list li {{
      margin: 0.15rem 0;
    }}
    tr.lang-merged-row td {{
      background: #fffbeb;
    }}
    .lang-stack-desc {{
      margin: 0 0 0.5rem;
      font-size: 0.78rem;
      color: var(--ink-muted);
    }}
    .lang-stack-chart {{
      margin-top: 0.25rem;
    }}
    .lang-stack-row {{
      display: grid;
      grid-template-columns: minmax(88px, 24%) 1fr auto;
      gap: 0.45rem;
      align-items: center;
      margin-bottom: 0.42rem;
      font-size: 0.78rem;
    }}
    .lang-stack-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
      color: var(--ink);
    }}
    .lang-stack-track {{
      display: flex;
      height: 14px;
      border-radius: 4px;
      overflow: hidden;
      background: #e2e8f0;
      min-width: 0;
      border: 1px solid #dbeafe;
    }}
    .lang-stack-seg {{
      min-width: 2px;
      height: 100%;
    }}
    .lang-stack-seg:hover {{
      filter: brightness(1.08);
      outline: 1px solid rgba(15, 23, 42, 0.25);
      z-index: 1;
    }}
    .lang-stack-singleton {{
      background: #cbd5e1;
    }}
    .lang-stack-total {{
      font-feature-settings: "tnum";
      color: var(--ink-muted);
      min-width: 2.5rem;
      text-align: right;
      white-space: nowrap;
    }}
    .lang-stack-legend {{
      margin: 0.55rem 0 0;
      font-size: 0.74rem;
      color: var(--ink-muted);
      display: flex;
      align-items: center;
      gap: 0.35rem;
      flex-wrap: wrap;
    }}
    .lang-stack-swatch {{
      display: inline-block;
      width: 0.75rem;
      height: 0.75rem;
      border-radius: 2px;
      border: 1px solid #94a3b8;
      flex-shrink: 0;
    }}
    .lang-len-section {{
      margin-top: 1.25rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }}
    .lang-len-section h3 {{
      margin: 0 0 0.35rem;
      font-size: 1rem;
    }}
    .lang-len-section h4 {{
      margin: 1rem 0 0.45rem;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--ink);
    }}
    .lang-len-median-chart {{
      margin-top: 0.25rem;
    }}
    .lang-len-median-track-wrap {{
      position: relative;
      min-width: 0;
    }}
    .lang-len-median-ref {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      margin-left: -1px;
      background: #ef4444;
      opacity: 0.85;
      pointer-events: none;
      z-index: 2;
    }}
    .lang-len-median-row {{
      display: grid;
      grid-template-columns: minmax(88px, 24%) 1fr auto;
      gap: 0.45rem;
      align-items: center;
      margin-bottom: 0.42rem;
      font-size: 0.78rem;
    }}
    .lang-len-median-track {{
      height: 14px;
      border-radius: 4px;
      background: #e2e8f0;
      border: 1px solid #dbeafe;
      overflow: hidden;
    }}
    .lang-len-median-fill {{
      height: 100%;
      background: #0ea5e9;
      border-radius: 3px;
      min-width: 2px;
    }}
    .lang-len-median-fill:hover {{
      filter: brightness(1.06);
    }}
    .lang-len-median-val {{
      font-feature-settings: "tnum";
      color: var(--ink-muted);
      min-width: 5.5rem;
      text-align: right;
      white-space: nowrap;
    }}
    .lang-len-note {{
      margin: 0.55rem 0 0;
      font-size: 0.74rem;
      color: var(--ink-muted);
    }}
    .lang-len-buckets-wrap {{
      margin-top: 0.25rem;
    }}
    .lang-len-bucket-toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.5rem 1rem;
      margin-bottom: 0.45rem;
    }}
    .lang-len-bucket-toolbar .lang-stack-desc {{
      margin: 0;
      flex: 1 1 16rem;
    }}
    .lang-len-seg {{
      display: inline-flex;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      overflow: hidden;
      flex-shrink: 0;
    }}
    .lang-len-seg-opt {{
      border: 0;
      background: #fff;
      color: var(--ink-muted);
      font-size: 0.72rem;
      padding: 0.28rem 0.55rem;
      cursor: pointer;
    }}
    .lang-len-seg-opt[aria-selected="true"] {{
      background: #0f172a;
      color: #fff;
    }}
    .lang-len-bucket-chart {{
      margin-top: 0.15rem;
    }}
    .lang-len-bucket-row {{
      display: grid;
      grid-template-columns: minmax(88px, 24%) 1fr auto;
      gap: 0.45rem;
      align-items: center;
      margin-bottom: 0.42rem;
      font-size: 0.78rem;
    }}
    .lang-len-bucket-bar-area {{
      min-width: 0;
    }}
    .lang-len-bucket-track {{
      display: flex;
      height: 14px;
      border-radius: 4px;
      overflow: hidden;
      background: #e2e8f0;
      border: 1px solid #dbeafe;
      min-width: 0;
    }}
    .lang-len-bucket-seg {{
      min-width: 2px;
      height: 100%;
    }}
    .lang-len-bucket-seg:hover {{
      filter: brightness(1.08);
      outline: 1px solid rgba(15, 23, 42, 0.25);
      z-index: 1;
    }}
    .lang-len-bucket-seg.is-short {{ background: #94a3b8; }}
    .lang-len-bucket-seg.is-medium {{ background: #f59e0b; }}
    .lang-len-bucket-seg.is-long {{ background: #6366f1; }}
    .lang-len-swatch-short {{ background: #94a3b8; }}
    .lang-len-swatch-medium {{ background: #f59e0b; }}
    .lang-len-swatch-long {{ background: #6366f1; }}
    .lang-verb-section {{
      margin-top: 1.25rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }}
    .lang-verb-section h3 {{
      margin: 0 0 0.35rem;
      font-size: 1rem;
    }}
    .lang-verb-section a {{
      color: #0369a1;
    }}
    .lang-verb-chart {{
      display: grid;
      grid-template-columns: 2.75rem 1fr;
      gap: 0.5rem;
      margin-top: 0.65rem;
    }}
    .lang-verb-yaxis {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      height: 240px;
      font-size: 0.68rem;
      color: var(--ink-muted);
      text-align: right;
      padding: 0 0.15rem 1.35rem 0;
      font-feature-settings: "tnum";
    }}
    .lang-verb-plot {{
      min-width: 0;
      border-left: 1px solid #cbd5e1;
      border-bottom: 1px solid #cbd5e1;
      padding: 0 0.25rem 0;
    }}
    .lang-verb-bars {{
      display: flex;
      align-items: flex-end;
      gap: 0.18rem;
      height: 240px;
      padding-bottom: 0;
    }}
    .lang-verb-col {{
      flex: 1 1 0;
      min-width: 0;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-end;
      gap: 0.25rem;
    }}
    .lang-verb-bar-wrap {{
      flex: 1 1 auto;
      width: 100%;
      display: flex;
      align-items: flex-end;
      justify-content: center;
      min-height: 0;
    }}
    .lang-verb-bar {{
      width: 78%;
      min-height: 2px;
      border-radius: 3px 3px 0 0;
      transition: filter 0.12s ease;
    }}
    .lang-verb-col:hover .lang-verb-bar {{
      filter: brightness(1.08);
      outline: 1px solid rgba(15, 23, 42, 0.2);
    }}
    .lang-verb-label {{
      flex: 0 0 auto;
      font-size: 0.62rem;
      line-height: 1.1;
      color: var(--ink-muted);
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: center;
      padding-bottom: 0.15rem;
    }}
    .lang-len-boxplot-chart {{
      margin-top: 0.25rem;
    }}
    .lang-len-box-row {{
      display: grid;
      grid-template-columns: minmax(88px, 24%) 1fr auto;
      gap: 0.45rem;
      align-items: center;
      margin-bottom: 0.42rem;
      font-size: 0.78rem;
    }}
    .lang-len-box-track-wrap {{
      position: relative;
      min-width: 0;
    }}
    .lang-len-box-track {{
      position: relative;
      height: 22px;
      border-radius: 4px;
      background: #f8fafc;
      border: 1px solid #dbeafe;
    }}
    .lang-len-box-ref {{
      position: absolute;
      top: -2px;
      bottom: -2px;
      width: 2px;
      margin-left: -1px;
      background: #ef4444;
      opacity: 0.85;
      pointer-events: none;
      z-index: 4;
    }}
    .lang-len-box-whisker {{
      position: absolute;
      top: 50%;
      height: 2px;
      margin-top: -1px;
      background: #64748b;
      z-index: 1;
    }}
    .lang-len-box-cap {{
      position: absolute;
      top: 50%;
      width: 2px;
      height: 10px;
      margin-top: -5px;
      margin-left: -1px;
      background: #64748b;
      z-index: 2;
    }}
    .lang-len-box-rect {{
      position: absolute;
      top: 4px;
      bottom: 4px;
      background: rgba(14, 165, 233, 0.35);
      border: 1px solid #0ea5e9;
      border-radius: 2px;
      z-index: 2;
    }}
    .lang-len-box-median {{
      position: absolute;
      top: 3px;
      bottom: 3px;
      width: 2px;
      margin-left: -1px;
      background: #0f172a;
      z-index: 3;
    }}
    .lang-len-box-outlier {{
      position: absolute;
      top: 50%;
      width: 6px;
      height: 6px;
      margin-top: -3px;
      margin-left: -3px;
      border-radius: 50%;
      background: #6366f1;
      border: 1px solid #4338ca;
      z-index: 3;
    }}
    .lang-len-box-axis-row {{
      display: grid;
      grid-template-columns: minmax(88px, 24%) 1fr auto;
      gap: 0.45rem;
      align-items: flex-start;
      margin-top: 0.15rem;
      font-size: 0.68rem;
      color: var(--ink-muted);
    }}
    .lang-len-box-axis-label {{
      font-weight: 500;
    }}
    .lang-len-box-axis {{
      position: relative;
      height: 1rem;
      min-width: 0;
    }}
    .lang-len-box-tick {{
      position: absolute;
      transform: translateX(-50%);
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Internal dataset report</h1>
      <p>Source: {html.escape(str(data.get("dataset_repo", "")))}</p>
      <p>Generated: {html.escape(when)}</p>
      <span class="badge">Internal — not for public site</span>
    </header>

    <div class="summary">
      <div class="stat"><div class="val">{totals.get("operator_episodes", 0):,}</div><div class="lbl">Operator episodes</div></div>
      <div class="stat"><div class="val">{totals.get("annotations", 0):,}</div><div class="lbl">Annotations</div></div>
      <div class="stat"><div class="val">{totals.get("labs", 0):,}</div><div class="lbl">Labs</div></div>
      <div class="stat"><div class="val">{totals.get("distinct_operators", 0):,}</div><div class="lbl">Distinct operators</div></div>
      <div class="stat"><div class="val">{totals.get("distinct_annotators", 0):,}</div><div class="lbl">Distinct annotators</div></div>
    </div>

    {stats_summary_html}

    {dataset_html}

    {language_html}

    <h2 class="labs-section-head">Operators &amp; annotators by lab</h2>
    <p class="section-desc">{people_scope_note}</p>

    <div class="toolbar">
      <input id="lab-filter" type="search" placeholder="Filter labs…" autocomplete="off">
      <button type="button" id="expand-all">Expand all</button>
      <button type="button" id="collapse-all">Collapse all</button>
    </div>

    <div id="lab-sections">
      {labs_html}
    </div>
    <p id="no-match" class="no-match">No labs match your filter.</p>
  </div>

  <script>
  (function () {{
    const folds = Array.from(document.querySelectorAll('.lab-fold'));
    const filter = document.getElementById('lab-filter');
    const noMatch = document.getElementById('no-match');

    function setOpen(fold, open) {{
      fold.classList.toggle('is-open', open);
      const btn = fold.querySelector('.lab-toggle');
      const body = fold.querySelector('.lab-body');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      body.hidden = !open;
    }}

    folds.forEach(fold => {{
      fold.querySelector('.lab-toggle').addEventListener('click', () => {{
        setOpen(fold, !fold.classList.contains('is-open'));
      }});
    }});

    filter.addEventListener('input', () => {{
      const q = filter.value.trim().toLowerCase();
      let visible = 0;
      folds.forEach(fold => {{
        const show = !q || fold.dataset.lab.includes(q);
        fold.style.display = show ? '' : 'none';
        if (show) visible += 1;
      }});
      noMatch.style.display = visible ? 'none' : 'block';
    }});

    document.getElementById('expand-all').addEventListener('click', () => {{
      folds.forEach(f => {{ if (f.style.display !== 'none') setOpen(f, true); }});
    }});
    document.getElementById('collapse-all').addEventListener('click', () => {{
      folds.forEach(f => setOpen(f, false));
    }});
  }})();

  (function () {{
    const dataEl = document.getElementById('lang-lab-data');
    const detail = document.getElementById('lang-lab-detail');
    if (!dataEl || !detail) return;

    let labData = {{}};
    try {{
      labData = JSON.parse(dataEl.textContent || '{{}}');
    }} catch (e) {{
      return;
    }}

    const nameEl = document.getElementById('lang-lab-detail-name');
    const instCountEl = document.getElementById('lang-inst-count');
    const failCountEl = document.getElementById('lang-fail-count');
    const instRows = document.getElementById('lang-inst-rows');
    const failRows = document.getElementById('lang-fail-rows');
    const closeBtn = document.getElementById('lang-lab-detail-close');
    const labRows = Array.from(document.querySelectorAll('.lang-lab-row'));

    function pct(v) {{
      return (100 * v).toFixed(1) + '%';
    }}

    function fillRows(tbody, items, emptyLabel) {{
      tbody.innerHTML = '';
      if (!items || !items.length) {{
        const tr = document.createElement('tr');
        tr.className = 'empty-row';
        tr.innerHTML = '<td colspan="3">' + emptyLabel + '</td>';
        tbody.appendChild(tr);
        return;
      }}
      items.forEach(function (item) {{
        const tr = document.createElement('tr');
        const text = String(item.text || '');
        const eps = Number(item.episodes || 0);
        const share = Number(item.share || 0);
        tr.innerHTML =
          '<td>' + text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</td>' +
          '<td>' + eps.toLocaleString() + '</td>' +
          '<td>' + pct(share) + '</td>';
        tbody.appendChild(tr);
      }});
    }}

    function showLab(lab) {{
      const entry = labData[lab];
      if (!entry) return;
      const corpusDetail = document.getElementById('lang-corpus-detail');
      if (corpusDetail) corpusDetail.hidden = true;
      document.querySelectorAll('.lang-corpus-stat').forEach(function (el) {{
        el.classList.remove('is-selected');
      }});
      labRows.forEach(function (row) {{
        row.classList.toggle('is-selected', row.dataset.lab === lab);
      }});
      nameEl.textContent = lab;
      const instructions = entry.instructions || [];
      const failures = entry.failures || [];
      instCountEl.textContent = String(instructions.length);
      failCountEl.textContent = String(failures.length);
      fillRows(instRows, instructions, 'No task instructions for this lab.');
      fillRows(failRows, failures, 'No failure descriptions for this lab.');
      detail.hidden = false;
      detail.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    function hideLab() {{
      detail.hidden = true;
      labRows.forEach(function (row) {{ row.classList.remove('is-selected'); }});
    }}

    labRows.forEach(function (row) {{
      row.addEventListener('click', function () {{ showLab(row.dataset.lab); }});
      row.addEventListener('keydown', function (e) {{
        if (e.key === 'Enter' || e.key === ' ') {{
          e.preventDefault();
          showLab(row.dataset.lab);
        }}
      }});
    }});

    if (closeBtn) closeBtn.addEventListener('click', hideLab);
  }})();

  (function () {{
    const dataEl = document.getElementById('lang-corpus-data');
    const detail = document.getElementById('lang-corpus-detail');
    if (!dataEl || !detail) return;

    let corpusData = {{}};
    try {{
      corpusData = JSON.parse(dataEl.textContent || '{{}}');
    }} catch (e) {{
      return;
    }}

    const titleEl = document.getElementById('lang-corpus-detail-title');
    const countEl = document.getElementById('lang-corpus-detail-count');
    const descEl = document.getElementById('lang-corpus-detail-desc');
    const headEl = document.getElementById('lang-corpus-head');
    const rowsEl = document.getElementById('lang-corpus-rows');
    const closeBtn = document.getElementById('lang-corpus-detail-close');
    const statEls = Array.from(document.querySelectorAll('.lang-corpus-stat'));
    const labDetail = document.getElementById('lang-lab-detail');
    let activeView = null;

    function esc(text) {{
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }}

    function pct(v) {{
      return (100 * v).toFixed(1) + '%';
    }}

    function clearStatSelection() {{
      statEls.forEach(function (el) {{ el.classList.remove('is-selected'); }});
      activeView = null;
    }}

    function hideCorpus() {{
      detail.hidden = true;
      clearStatSelection();
    }}

    function memberListHtml(members) {{
      if (!members || !members.length) return '';
      const items = members.map(function (member) {{
        const label = esc(member.text || member.normalized || '');
        const eps = Number(member.episodes || 0).toLocaleString();
        return '<li>' + label + ' <span class="lang-variant-note">(' + eps + ' ep)</span></li>';
      }}).join('');
      return '<ul class="lang-member-list">' + items + '</ul>';
    }}

    function showNormalized() {{
      const items = corpusData.instructions_normalized || [];
      const threshold = Number(corpusData.near_duplicate_threshold || 0.9);
      headEl.innerHTML = '<tr><th>Instruction</th><th>Episodes</th><th>Share</th></tr>';
      titleEl.textContent = 'All unique task instructions';
      countEl.textContent = String(items.length);
      descEl.textContent =
        'One row per normalized form (case and trailing punctuation ignored). ' +
        'Shows the most frequent exact wording. Near-dup grouping merges forms with ≥' +
        Math.round(threshold * 100) + '% character similarity (see Near-dup groups).';

      rowsEl.innerHTML = '';
      if (!items.length) {{
        rowsEl.innerHTML = '<tr class="empty-row"><td colspan="3">No task instructions.</td></tr>';
        return;
      }}
      items.forEach(function (item) {{
        const tr = document.createElement('tr');
        const text = String(item.text || '');
        const variants = item.variants || [];
        let cell = esc(text);
        if (variants.length) {{
          cell += '<span class="lang-variant-note" title="' +
            esc(variants.join('\\n')) + '">+' + variants.length +
            ' exact variant' + (variants.length === 1 ? '' : 's') + '</span>';
        }}
        tr.innerHTML =
          '<td>' + cell + '</td>' +
          '<td>' + Number(item.episodes || 0).toLocaleString() + '</td>' +
          '<td>' + pct(Number(item.share || 0)) + '</td>';
        rowsEl.appendChild(tr);
      }});
    }}

    function showNearDuplicate() {{
      const groups = corpusData.instructions_near_duplicate_groups || [];
      const threshold = Number(corpusData.near_duplicate_threshold || 0.9);
      const normalizedUnique = Number(corpusData.normalized_unique || 0);
      const merges = Number(corpusData.near_duplicate_merges || 0);
      const singletons = groups.filter(function (g) {{ return Number(g.member_count || 0) <= 1; }}).length;

      headEl.innerHTML = '<tr><th>Representative</th><th>Members</th><th>Episodes</th><th>Share</th></tr>';
      titleEl.textContent = 'Near-duplicate groups';
      countEl.textContent = String(groups.length);
      descEl.textContent =
        'Start from ' + normalizedUnique.toLocaleString() + ' normalized strings. ' +
        'Greedy clustering: sort strings alphabetically; each joins the first group whose representative ' +
        'has ≥' + Math.round(threshold * 100) + '% Python SequenceMatcher character similarity. ' +
        merges.toLocaleString() + ' group(s) merge 2+ strings; ' +
        singletons.toLocaleString() + ' group(s) contain a single string.';

      rowsEl.innerHTML = '';
      if (!groups.length) {{
        rowsEl.innerHTML = '<tr class="empty-row"><td colspan="4">No groups.</td></tr>';
        return;
      }}
      groups.forEach(function (group) {{
        const tr = document.createElement('tr');
        if (Number(group.member_count || 0) > 1) tr.className = 'lang-merged-row';
        const members = group.members || [];
        tr.innerHTML =
          '<td>' + esc(group.representative || '') + '</td>' +
          '<td>' + memberListHtml(members) + '</td>' +
          '<td>' + Number(group.episodes || 0).toLocaleString() + '</td>' +
          '<td>' + pct(Number(group.share || 0)) + '</td>';
        rowsEl.appendChild(tr);
      }});
    }}

    function showCorpusView(view, corpusId) {{
      if (corpusId !== 'instructions') return;
      statEls.forEach(function (el) {{
        el.classList.toggle(
          'is-selected',
          el.dataset.corpus === corpusId && el.dataset.view === view
        );
      }});
      if (labDetail) labDetail.hidden = true;
      activeView = view;
      if (view === 'near_duplicate') showNearDuplicate();
      else showNormalized();
      detail.hidden = false;
      detail.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    statEls.forEach(function (el) {{
      el.addEventListener('click', function () {{
        const view = el.dataset.view;
        const selected = el.classList.contains('is-selected') && !detail.hidden && activeView === view;
        if (selected) {{
          hideCorpus();
          return;
        }}
        showCorpusView(view, el.dataset.corpus);
      }});
      el.addEventListener('keydown', function (e) {{
        if (e.key === 'Enter' || e.key === ' ') {{
          e.preventDefault();
          showCorpusView(el.dataset.view, el.dataset.corpus);
        }}
      }});
    }});

    if (closeBtn) closeBtn.addEventListener('click', hideCorpus);
  }})();

  (function () {{
    const dataEl = document.getElementById('lang-fail-length-data');
    const container = document.getElementById('lang-len-bucket-chart');
    const descEl = document.getElementById('lang-len-bucket-desc');
    const seg = document.getElementById('lang-len-bucket-seg');
    if (!dataEl || !container) return;

    let perLab = {{}};
    try {{
      perLab = JSON.parse(dataEl.textContent || '{{}}');
    }} catch (e) {{
      return;
    }}

    let mode = 'rate';

    function esc(text) {{
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }}

    function renderLengthBuckets(viewMode) {{
      mode = viewMode || mode || 'rate';
      const labs = Object.keys(perLab).sort(function (a, b) {{
        return (perLab[b].count || 0) - (perLab[a].count || 0);
      }});

      if (!labs.length) {{
        container.innerHTML = '<p class="empty">No described failures.</p>';
        container.className = 'lang-len-bucket-chart is-' + mode;
        return;
      }}

      const maxTotal = labs.reduce(function (max, lab) {{
        return Math.max(max, Number((perLab[lab] || {{}}).count || 0));
      }}, 1);

      const bucketOrder = ['short', 'medium', 'long'];
      const bucketClass = {{
        short: 'is-short',
        medium: 'is-medium',
        long: 'is-long',
      }};

      const rows = labs.map(function (lab) {{
        const stats = perLab[lab] || {{}};
        const total = Number(stats.count || 0);
        const buckets = stats.buckets || {{}};
        const trackWidth = mode === 'volume'
          ? Math.max(4, Math.round(1000 * total / maxTotal) / 10)
          : 100;
        const segments = bucketOrder.map(function (bucket) {{
          const info = buckets[bucket] || {{}};
          const count = Number(info.count || 0);
          if (count <= 0) return '';
          const share = Number(info.share || 0);
          const pctLabel = (100 * share).toFixed(1);
          const title = bucket.charAt(0).toUpperCase() + bucket.slice(1) +
            ': ' + count.toLocaleString() + ' (' + pctLabel + '%)';
          return (
            '<span class="lang-len-bucket-seg ' + bucketClass[bucket] + '" style="flex:' +
            count + ' 0 0" title="' + esc(title) + '"></span>'
          );
        }}).join('');

        return (
          '<div class="lang-len-bucket-row">' +
            '<span class="lang-stack-label" title="' + esc(lab) + '">' + esc(lab) + '</span>' +
            '<div class="lang-len-bucket-bar-area">' +
              '<div class="lang-len-bucket-track" style="width:' + trackWidth + '%">' +
                segments +
              '</div>' +
            '</div>' +
            '<span class="lang-stack-total">' + total.toLocaleString() + '</span>' +
          '</div>'
        );
      }});

      container.className = 'lang-len-bucket-chart is-' + mode;
      container.innerHTML = rows.join('');
    }}

    function setLengthMode(nextMode) {{
      mode = nextMode;
      if (descEl) {{
        descEl.textContent = mode === 'volume'
          ? 'Bar length is proportional to described-failure count. Colors = short / medium / long word buckets.'
          : 'Each bar spans the full width so you can compare short / medium / long shares across labs.';
      }}
      renderLengthBuckets(mode);
      if (seg) {{
        seg.querySelectorAll('.lang-len-seg-opt').forEach(function (btn) {{
          btn.setAttribute('aria-selected', btn.getAttribute('data-mode') === mode ? 'true' : 'false');
        }});
      }}
    }}

    if (seg) {{
      seg.querySelectorAll('.lang-len-seg-opt').forEach(function (btn) {{
        btn.addEventListener('click', function () {{
          const nextMode = btn.getAttribute('data-mode');
          if (!nextMode || nextMode === mode) return;
          setLengthMode(nextMode);
        }});
      }});
    }}

    setLengthMode('rate');
  }})();
  </script>
</body>
</html>
"""


def render_internal_report(data: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_internal_report_html(data), encoding="utf-8")


def load_internal_stats(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
