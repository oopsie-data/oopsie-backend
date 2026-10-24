"""Tier 1 lexical diversity metrics for task instructions and failure descriptions."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

TOP_STRINGS_LIMIT = 15
NEAR_DUPLICATE_RATIO = 0.9
FAILURE_LENGTH_SHORT_MAX_WORDS = 5
FAILURE_LENGTH_MEDIUM_MAX_WORDS = 15
VERB_CHART_TOP_N = 20

_INSTRUCTION_STOPWORDS = frozenset(
    """
    the a an this that these those please robot you your then and to in on at with from
    into using use left right first next all one two three it its there here of for by is
    are be after before while when if so also just only each every both either or not do
    does can should will would could must may might have has had been being was were am i
    we they he she my our their his her me us them up down out off over under away back
    around through onto upon about as than but once again still already now very more most
    some any no other such same new old leftmost rightmost top bottom middle front rear
    near far closest farthest nearest largest smallest second third fourth fifth last
    final initial starting end way part side hand hands gripper arm object item thing
    """.split()
)

_INSTRUCTION_LIGHT_VERBS = frozenset(
    {
        "perform",
        "complete",
        "execute",
        "try",
        "attempt",
        "help",
        "start",
        "continue",
        "finish",
        "go",
        "get",
        "use",
        "make",
        "do",
        "be",
    }
)

_VERB_IRREGULAR = {
    "picked": "pick",
    "picking": "pick",
    "puts": "put",
    "putting": "put",
    "placed": "place",
    "placing": "place",
    "moved": "move",
    "moving": "move",
    "removed": "remove",
    "removing": "remove",
    "pushed": "push",
    "pushing": "push",
    "stacked": "stack",
    "stacking": "stack",
    "closed": "close",
    "closing": "close",
    "opened": "open",
    "opening": "open",
    "hung": "hang",
    "hanging": "hang",
    "took": "take",
    "taking": "take",
    "unstacked": "unstack",
    "unstacking": "unstack",
    "brought": "bring",
    "bringing": "bring",
    "found": "find",
    "finding": "find",
    "went": "go",
    "going": "go",
    "done": "do",
    "doing": "do",
    "did": "do",
}

_PHRASAL_VERB_PREFIXES: tuple[tuple[str, str], ...] = (
    ("pick up", "pick"),
    ("put down", "place"),
    ("take out", "take"),
    ("take off", "remove"),
    ("take down", "remove"),
    ("turn on", "turn"),
    ("turn off", "turn"),
    ("hang on", "hang"),
    ("hang up", "hang"),
    ("put away", "put"),
    ("lift up", "lift"),
    ("pull out", "pull"),
    ("push in", "push"),
    ("slide out", "slide"),
    ("slide in", "slide"),
    ("set up", "set"),
)

_TOKEN_RE = re.compile(r"[a-z']+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    normalized = unicodedata.normalize("NFKC", text.strip()).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.rstrip(".,!?;:")


def _lemmatize_verb(token: str) -> str:
    if token in _VERB_IRREGULAR:
        return _VERB_IRREGULAR[token]
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if len(stem) >= 3:
            return stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if stem.endswith("i"):
            stem = f"{stem[:-1]}y"
        elif len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if len(stem) >= 3:
            return stem
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("es") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 3:
            return stem
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def extract_instruction_verb(text: str) -> str | None:
    """Heuristic primary verb for robotics task instructions (DROID-style skill label)."""
    normalized = normalize_text(text)
    if not normalized:
        return None
    if "layup" in normalized:
        return "layup"

    for phrase, verb in _PHRASAL_VERB_PREFIXES:
        if normalized == phrase or normalized.startswith(f"{phrase} "):
            return verb

    tokens = _TOKEN_RE.findall(normalized)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token in _INSTRUCTION_STOPWORDS:
            continue
        if token in _INSTRUCTION_LIGHT_VERBS and index < len(tokens):
            continue
        lemma = _lemmatize_verb(token)
        if len(lemma) >= 2:
            return lemma
    return None


def _summarize_verb_distribution(
    verb_counts: Counter[str],
    *,
    total_instruction_episodes: int,
    top_n: int = VERB_CHART_TOP_N,
) -> dict[str, Any]:
    ranked = verb_counts.most_common()
    if not ranked:
        return {
            "method": "first_content_verb",
            "total_episodes": int(total_instruction_episodes),
            "classified_episodes": 0,
            "unique_verbs": 0,
            "verbs": [],
        }

    if len(ranked) > top_n:
        top = ranked[:top_n]
        other_count = sum(count for _, count in ranked[top_n:])
        if other_count:
            top.append(("other", other_count))
    else:
        top = ranked

    classified = sum(verb_counts.values())
    return {
        "method": "first_content_verb",
        "total_episodes": int(total_instruction_episodes),
        "classified_episodes": int(classified),
        "unique_verbs": len(verb_counts),
        "verbs": [
            {
                "verb": verb,
                "episodes": int(count),
                "share": round(count / total_instruction_episodes, 4)
                if total_instruction_episodes
                else 0.0,
            }
            for verb, count in top
        ],
    }


def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(pct / 100 * len(ordered))) - 1))
    return float(ordered[idx])


def _failure_length_bucket(word_count: int) -> str:
    if word_count <= FAILURE_LENGTH_SHORT_MAX_WORDS:
        return "short"
    if word_count <= FAILURE_LENGTH_MEDIUM_MAX_WORDS:
        return "medium"
    return "long"


def _box_plot_stats(words: list[int]) -> dict[str, float | int]:
    """Tukey box summary: Q1/median/Q3, whiskers at 1.5×IQR fences, outlier count."""
    if not words:
        return {}
    ordered = sorted(words)
    q1 = _percentile(words, 25)
    median = _percentile(words, 50)
    q3 = _percentile(words, 75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    inner = [w for w in ordered if fence_low <= w <= fence_high]
    whisker_low = float(min(inner)) if inner else float(ordered[0])
    whisker_high = float(max(inner)) if inner else float(ordered[-1])
    outliers = [w for w in ordered if w < fence_low or w > fence_high]
    return {
        "min_words": float(ordered[0]),
        "max_words": float(ordered[-1]),
        "p25_words": round(q1, 1),
        "median_words": round(median, 1),
        "p75_words": round(q3, 1),
        "whisker_low_words": whisker_low,
        "whisker_high_words": whisker_high,
        "outlier_count": len(outliers),
    }


def _summarize_failure_length_distribution(
    failure_words_by_lab: dict[str, list[int]],
) -> dict[str, Any]:
    """Per-lab word-length stats among described failures (one count per failure episode)."""
    all_words: list[int] = []
    per_lab: dict[str, Any] = {}
    for lab in sorted(failure_words_by_lab):
        words = failure_words_by_lab[lab]
        if not words:
            continue
        buckets: Counter[str] = Counter(_failure_length_bucket(w) for w in words)
        total = len(words)
        per_lab[lab] = {
            "count": total,
            **_box_plot_stats(words),
            "buckets": {
                bucket: {
                    "count": int(buckets[bucket]),
                    "share": round(buckets[bucket] / total, 4),
                }
                for bucket in ("short", "medium", "long")
            },
        }
        all_words.extend(words)

    corpus_box = _box_plot_stats(all_words)
    return {
        "bucket_thresholds": {
            "short_max_words": FAILURE_LENGTH_SHORT_MAX_WORDS,
            "medium_max_words": FAILURE_LENGTH_MEDIUM_MAX_WORDS,
        },
        "corpus_median_words": corpus_box.get("median_words", 0.0),
        "corpus_count": len(all_words),
        "corpus_box": corpus_box,
        "per_lab": per_lab,
    }


def _shannon_entropy_bits(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _top_k_share(counts: Counter[str], total: int, k: int) -> float:
    if total <= 0:
        return 0.0
    return sum(c for _, c in counts.most_common(k)) / total


def _build_near_duplicate_groups(normalized_strings: list[str]) -> list[list[str]]:
    """Greedy clusters of normalized strings with SequenceMatcher ratio >= threshold."""
    groups: list[list[str]] = []
    for text in sorted(set(normalized_strings)):
        if not text:
            continue
        placed = False
        for group in groups:
            if SequenceMatcher(None, text, group[0]).ratio() >= NEAR_DUPLICATE_RATIO:
                group.append(text)
                placed = True
                break
        if not placed:
            groups.append([text])
    return groups


def _near_duplicate_groups(normalized_strings: list[str]) -> int:
    return len(_build_near_duplicate_groups(normalized_strings))


def _near_duplicate_group_list(
    normalized_weighted: Counter[str],
    normalized_rows: list[dict[str, Any]],
    total_episodes: int,
) -> list[dict[str, Any]]:
    """Export near-duplicate clusters with member strings and episode weights."""
    rows_by_norm = {row["normalized"]: row for row in normalized_rows}
    groups = _build_near_duplicate_groups(list(normalized_weighted.keys()))
    result: list[dict[str, Any]] = []
    for group in groups:
        members = [rows_by_norm[norm] for norm in group if norm in rows_by_norm]
        members.sort(key=lambda row: (-int(row["episodes"]), row["text"]))
        episodes = sum(int(row["episodes"]) for row in members)
        representative = members[0]["text"] if members else group[0]
        result.append(
            {
                "representative": representative,
                "member_count": len(members),
                "episodes": episodes,
                "share": round(episodes / total_episodes, 4) if total_episodes else 0.0,
                "members": members,
            }
        )
    result.sort(key=lambda row: (-row["episodes"], row["representative"]))
    return result


def _summarize_corpus(
    weighted_strings: Counter[str],
    *,
    total_episodes: int,
) -> dict[str, Any]:
    raw_unique = len(weighted_strings)
    normalized_weighted: Counter[str] = Counter()
    word_lengths: list[int] = []

    for text, count in weighted_strings.items():
        if not text.strip():
            continue
        norm = normalize_text(text)
        if not norm:
            continue
        normalized_weighted[norm] += count
        word_lengths.extend([_word_count(text)] * count)

    per_string_counts = list(weighted_strings.values())
    entropy = _shannon_entropy_bits(weighted_strings, total_episodes)

    top_strings = []
    for text, count in weighted_strings.most_common(TOP_STRINGS_LIMIT):
        top_strings.append(
            {
                "text": text,
                "episodes": int(count),
                "share": round(count / total_episodes, 4) if total_episodes else 0.0,
            }
        )

    all_strings = [
        {
            "text": text,
            "episodes": int(count),
            "share": round(count / total_episodes, 4) if total_episodes else 0.0,
        }
        for text, count in weighted_strings.most_common()
        if text.strip()
    ]
    normalized_strings = _normalized_string_list(weighted_strings, total_episodes)
    near_duplicate_group_list = _near_duplicate_group_list(
        normalized_weighted, normalized_strings, total_episodes
    )

    return {
        "total_episodes": int(total_episodes),
        "raw_unique": int(raw_unique),
        "normalized_unique": int(len(normalized_weighted)),
        "near_duplicate_groups": len(near_duplicate_group_list),
        "near_duplicate_threshold": NEAR_DUPLICATE_RATIO,
        "near_duplicate_merges": sum(
            1 for group in near_duplicate_group_list if group["member_count"] > 1
        ),
        "entropy_bits": round(entropy, 3),
        "effective_vocabulary": round(2**entropy, 1),
        "concentration": {
            "top_5_share": round(_top_k_share(weighted_strings, total_episodes, 5), 4),
            "top_10_share": round(_top_k_share(weighted_strings, total_episodes, 10), 4),
            "top_20_share": round(_top_k_share(weighted_strings, total_episodes, 20), 4),
        },
        "episodes_per_string": {
            "median": _percentile(per_string_counts, 50),
            "p90": _percentile(per_string_counts, 90),
            "max": max(per_string_counts) if per_string_counts else 0,
        },
        "length_words": {
            "median": _percentile(word_lengths, 50),
            "p90": _percentile(word_lengths, 90),
        },
        "top_strings": top_strings,
        "all_strings": all_strings,
        "normalized_strings": normalized_strings,
        "near_duplicate_group_list": near_duplicate_group_list,
    }


def _normalized_string_list(
    weighted_strings: Counter[str], total_episodes: int
) -> list[dict[str, Any]]:
    """One row per normalized form; canonical text is the most frequent exact variant."""
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for text, count in weighted_strings.items():
        norm = normalize_text(text)
        if not norm:
            continue
        groups[norm].append((text, int(count)))

    rows: list[dict[str, Any]] = []
    for norm, variants in groups.items():
        variants.sort(key=lambda item: (-item[1], item[0]))
        canonical, _ = variants[0]
        episodes = sum(count for _, count in variants)
        row: dict[str, Any] = {
            "text": canonical,
            "normalized": norm,
            "episodes": episodes,
            "share": round(episodes / total_episodes, 4) if total_episodes else 0.0,
        }
        if len(variants) > 1:
            row["variants"] = [text for text, _ in variants[1:]]
        rows.append(row)
    rows.sort(key=lambda item: (-item["episodes"], item["text"]))
    return rows


def _per_lab_string_lists(
    lab_strings: dict[str, Counter[str]],
    lab_totals: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    per_lab: dict[str, list[dict[str, Any]]] = {}
    for lab in sorted(lab_strings):
        weighted = lab_strings[lab]
        total = int(lab_totals[lab])
        per_lab[lab] = [
            {
                "text": text,
                "episodes": int(count),
                "share": round(count / total, 4) if total else 0.0,
            }
            for text, count in weighted.most_common()
        ]
    return per_lab


def _summarize_per_lab(
    lab_strings: dict[str, Counter[str]],
    lab_totals: Counter[str],
) -> dict[str, Any]:
    per_lab: dict[str, Any] = {}
    for lab in sorted(lab_strings):
        weighted = lab_strings[lab]
        total = int(lab_totals[lab])
        normalized = {normalize_text(t) for t in weighted if t.strip()}
        entropy = _shannon_entropy_bits(weighted, total)
        per_lab[lab] = {
            "episodes": total,
            "raw_unique": len(weighted),
            "normalized_unique": len(normalized),
            "entropy_bits": round(entropy, 3),
            "effective_vocabulary": round(2**entropy, 1),
            "top_5_share": round(_top_k_share(weighted, total, 5), 4),
        }
    return per_lab


def aggregate_language_diversity(
    records: list[dict],
    repo: str,
    *,
    include_excluded: bool = False,
    scope_label: str = "public_corpus",
) -> dict[str, Any]:
    """Compute Tier 1 diversity metrics from episode records."""
    from oopsie_backend.common.repos import (
        PUBLIC_EXCLUDED_DATASETS,
        STATS_EXCLUDED_DATASETS,
    )
    from oopsie_backend.stats.episode_stats import (
        _include_in_any_stats,
        _include_in_public_stats,
        _primary_complete_annotation,
    )

    instruction_weighted: Counter[str] = Counter()
    instruction_by_lab: dict[str, Counter[str]] = defaultdict(Counter)
    instruction_lab_totals: Counter[str] = Counter()
    instruction_verb_counts: Counter[str] = Counter()
    instruction_episodes = 0

    failure_weighted: Counter[str] = Counter()
    failure_by_lab: dict[str, Counter[str]] = defaultdict(Counter)
    failure_lab_totals: Counter[str] = Counter()
    failure_words_by_lab: dict[str, list[int]] = defaultdict(list)

    total_episodes = 0
    total_failures = 0
    failures_with_description = 0

    for record in records:
        if not _include_in_any_stats(record, repo):
            continue
        if not include_excluded and not _include_in_public_stats(record, repo):
            continue

        lab = record.get("lab_id") or "unknown_lab"
        total_episodes += 1
        instruction_lab_totals[lab] += 1

        instruction = (record.get("language_instruction") or "").strip()
        if instruction:
            instruction_episodes += 1
            instruction_weighted[instruction] += 1
            instruction_by_lab[lab][instruction] += 1
            verb = extract_instruction_verb(instruction)
            if verb:
                instruction_verb_counts[verb] += 1

        primary = _primary_complete_annotation(record.get("annotations") or [])
        if primary is None or primary["success"] >= 0.5:
            continue

        total_failures += 1
        description = (primary.get("failure_description") or "").strip()
        if not description:
            continue

        failures_with_description += 1
        failure_weighted[description] += 1
        failure_by_lab[lab][description] += 1
        failure_lab_totals[lab] += 1
        failure_words_by_lab[lab].append(_word_count(description))

    return {
        "scope": scope_label,
        "excluded_repos": sorted(PUBLIC_EXCLUDED_DATASETS) if not include_excluded else [],
        "excluded_from_all_stats_datasets": sorted(STATS_EXCLUDED_DATASETS),
        "task_instructions": {
            **_summarize_corpus(instruction_weighted, total_episodes=total_episodes),
            "per_lab": _summarize_per_lab(instruction_by_lab, instruction_lab_totals),
            "per_lab_strings": _per_lab_string_lists(
                instruction_by_lab, instruction_lab_totals
            ),
            "verb_distribution": _summarize_verb_distribution(
                instruction_verb_counts,
                total_instruction_episodes=instruction_episodes,
            ),
        },
        "failure_descriptions": {
            "total_failures": int(total_failures),
            "with_description": int(failures_with_description),
            "description_coverage": round(
                failures_with_description / total_failures, 4
            )
            if total_failures
            else 0.0,
            **_summarize_corpus(
                failure_weighted,
                total_episodes=failures_with_description,
            ),
            "per_lab": _summarize_per_lab(failure_by_lab, failure_lab_totals),
            "per_lab_strings": _per_lab_string_lists(
                failure_by_lab, failure_lab_totals
            ),
            "length_distribution": _summarize_failure_length_distribution(
                failure_words_by_lab
            ),
        },
    }
