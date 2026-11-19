"""Krippendorff's α and the pairing of failure segments between annotators."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from itertools import permutations


def krippendorff_alpha(units: Sequence[Sequence[Hashable]], order: Sequence[Hashable] | None = None) -> float:
    """α over ``units``, each the values its raters gave; units with fewer than two values are ignored.

    Nominal distance by default; with ``order``, the ordinal distance over that order of values.
    """
    coincidences: Counter = Counter()
    for values in units:
        m = len(values)
        if m < 2:
            continue
        for a, b in permutations(values, 2):
            coincidences[(a, b)] += 1 / (m - 1)
    marginals: Counter = Counter()
    for (a, _), w in coincidences.items():
        marginals[a] += w
    n = sum(marginals.values())
    if n <= 1:
        return math.nan
    categories = list(order) if order is not None else sorted(marginals, key=str)
    if order is not None:
        rank = {c: i for i, c in enumerate(categories)}

        def delta(a, b) -> float:
            lo, hi = sorted((rank[a], rank[b]))
            between = sum(marginals[c] for c in categories[lo : hi + 1])
            return (between - (marginals[a] + marginals[b]) / 2) ** 2
    else:

        def delta(a, b) -> float:
            return 0.0 if a == b else 1.0

    observed = sum(w * delta(a, b) for (a, b), w in coincidences.items()) / n
    expected = sum(marginals[a] * marginals[b] * delta(a, b) for a in categories for b in categories) / (n * (n - 1))
    return 1.0 - observed / expected if expected else math.nan


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def pair_segments(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> list[tuple[int, int, float]]:
    """Greedy pairs ``(i, j, iou)`` of overlapping segments of two annotators, highest overlap first."""
    candidates = sorted(
        ((iou(sa, sb), i, j) for i, sa in enumerate(a) for j, sb in enumerate(b)), reverse=True
    )
    used_a, used_b, pairs = set(), set(), []
    for overlap, i, j in candidates:
        if overlap <= 0:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, overlap))
    return pairs


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, non-overlapping union of ``intervals``."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect_intervals(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intersection of two merged interval lists."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def union_iou(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    """IoU of the time two merged interval lists cover; 1 when both are empty."""
    if not a and not b:
        return 1.0
    inter = sum(e - s for s, e in intersect_intervals(a, b))
    total = sum(e - s for s, e in a) + sum(e - s for s, e in b) - inter
    return inter / total if total > 0 else 0.0


def jaccard(a: set, b: set) -> float:
    """Jaccard index of two sets; 1 when both are empty."""
    return len(a & b) / len(a | b) if a | b else 1.0
