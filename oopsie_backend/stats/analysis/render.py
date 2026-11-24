"""Render analysis reports (JSON) as Markdown.

Recognized shapes become tables: summaries (``n``, ``mean``, quantiles), distributions
(``n``, ``counts``, ``shares``), shares with a confidence interval (``share``, ``ci``),
groups of any of these, numeric matrices and lists of records. Anything else becomes
nested headings with a table of its scalar values.
"""

from __future__ import annotations

import json
from pathlib import Path

QUANTILES = ("q05", "q25", "q50", "q75", "q95")
MAX_ROWS = 25
MAX_COLUMNS = 12
MAX_CELL = 80


def _title(key: str) -> str:
    head, _, rest = key.partition("_")
    if head.isdigit():
        return f"{head}. {rest.replace('_', ' ').capitalize()}"
    return key.replace("_", " ")


def _is_share_key(key: str) -> bool:
    return key == "share" or key.endswith("_share") or key.startswith("share_")


def _num(value, percent: bool = False) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if percent:
            return f"{100 * value:.1f}%"
        return f"{value:,.1f}" if abs(value) >= 10 else f"{value:.3g}"
    if isinstance(value, list):
        return "[" + ", ".join(_num(v, percent) for v in value) + "]"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 1] + "…"


def _table(header: list[str], rows: list[list[str]], total: int | None = None) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[:MAX_ROWS]]
    total = len(rows) if total is None else total
    if total > MAX_ROWS:
        lines.append(f"\n*{total - MAX_ROWS} more rows not shown.*")
    return lines + [""]


def _is_summary(v) -> bool:
    return isinstance(v, dict) and {"n", "mean", "q50"} <= v.keys()


def _is_ci(v) -> bool:
    return isinstance(v, dict) and {"share", "ci"} <= v.keys()


def _is_counts(v) -> bool:
    return isinstance(v, dict) and "counts" in v and isinstance(v["counts"], dict)


def _is_ci_shares(v) -> bool:
    return (
        isinstance(v, dict)
        and isinstance(v.get("shares"), dict)
        and bool(v["shares"])
        and all(_is_ci(s) for s in v["shares"].values())
    )


def _is_below_min(v) -> bool:
    return isinstance(v, dict) and v.get("below_min") is True


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (int, float, str, bool)) or (
        isinstance(v, list) and len(v) <= 5 and all(isinstance(x, (int, float, str)) for x in v)
    )


def _uniform(d: dict, test) -> bool:
    values = [v for v in d.values() if not _is_below_min(v)]
    return bool(values) and all(test(v) for v in values)


def _below_min_note(d: dict) -> list[str]:
    small = [f"{k} (n={v.get('n', '?')})" for k, v in d.items() if _is_below_min(v)]
    return [f"*Too few episodes to compare:* {', '.join(small)}.", ""] if small else []


def _summary_table(groups: dict, label: str) -> list[str]:
    rows = [[_num(k), _num(v["n"]), _num(v["mean"]), *(_num(v.get(q)) for q in QUANTILES)] for k, v in groups.items()]
    return _table([label, "n", "mean", *QUANTILES], rows)


def _counts_table(d: dict) -> list[str]:
    shares = d.get("shares", {})
    rows = [[_num(k), _num(c), _num(shares.get(k), percent=True)] for k, c in d["counts"].items()]
    lines = [f"n = {_num(d['n'])}", ""] if "n" in d else []
    return lines + _table(["value", "count", "share"], rows)


def _ci_table(d: dict) -> list[str]:
    rows = [[_num(k), _num(s.get("count")), _num(s["share"], True), _num(s["ci"], True)] for k, s in d["shares"].items()]
    lines = [f"n = {_num(d['n'])}", ""] if "n" in d else []
    return lines + _table(["value", "count", "share", "95% CI"], rows)


def _group_matrix(groups: dict, label: str, cells: str) -> list[str]:
    """Groups by values: one row per group with its n, one column per value, cells are shares."""
    shown = {k: v for k, v in groups.items() if not _is_below_min(v)}
    totals: dict[str, float] = {}
    for v in shown.values():
        for k, s in v[cells].items():
            totals[k] = totals.get(k, 0) + (s["count"] if _is_ci(s) else v.get("counts", {}).get(k, s))
    columns = sorted(totals, key=totals.get, reverse=True)[:MAX_COLUMNS]
    rows = []
    for g, v in shown.items():
        row = [_num(g), _num(v.get("n"))]
        for c in columns:
            s = v[cells].get(c)
            row.append(_num(s["share"] if _is_ci(s) else s, percent=True) if s is not None else "–")
        rows.append(row)
    lines = _table([label, "n", *columns], rows)
    if len(totals) > MAX_COLUMNS:
        lines.insert(-1, f"\n*{len(totals) - MAX_COLUMNS} rarer values not shown.*")
    return lines + _below_min_note(groups)


def _shared_values(groups: dict, cells: str) -> bool:
    keys = [set(v[cells]) for v in groups.values() if not _is_below_min(v)]
    if len(keys) < 2:
        return True
    union = set().union(*keys)
    shared = sum(1 for k in union if sum(k in s for s in keys) >= 2)
    return shared >= 0.5 * len(union)


def _top(d: dict, n: int = 5) -> str:
    shares = d.get("shares") or {}
    items = sorted(d["counts"].items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{k} {_num(shares[k], True)}" if k in shares else f"{k} {_num(c)}" for k, c in items)


def _record_groups(groups: dict) -> bool:
    """Groups whose fields are scalars or distributions, e.g. per hardware: episodes, labs, verbs."""
    values = [v for v in groups.values() if not _is_below_min(v)]
    return _uniform(groups, lambda v: isinstance(v, dict) and all(_is_scalar(x) or _is_counts(x) for x in v.values())) and all(
        v.keys() == values[0].keys() for v in values
    )


def _record_groups_table(groups: dict, label: str) -> list[str]:
    shown = {k: v for k, v in groups.items() if not _is_below_min(v)}
    columns: list[str] = []
    for v in shown.values():
        columns += [c for c in v if c not in columns]
    rows = []
    for g, v in shown.items():
        row = [_num(g)]
        for c in columns:
            x = v.get(c)
            row.append(_top(x) if _is_counts(x) else _num(x, _is_share_key(c)))
        rows.append(row)
    header = [label, *(f"top {c}" if any(_is_counts(v.get(c)) for v in shown.values()) else c for c in columns)]
    return _table(header, rows) + _below_min_note(groups)


def _ci_groups_table(groups: dict, label: str) -> list[str]:
    extra = [f for f in next(iter(groups.values())) if f not in ("share", "ci")]
    rows = [[_num(k), *(_num(v.get(f)) for f in extra), _num(v["share"], True), _num(v["ci"], True)] for k, v in groups.items()]
    return _table([label, *extra, "share", "95% CI"], rows)


def _pivot(groups: dict) -> bool:
    """Groups with the same nested fields, e.g. per annotator: outcomes, severity, categories."""
    values = list(groups.values())
    return (
        len(values) >= 2
        and all(isinstance(v, dict) and v for v in values)
        and all(v.keys() == values[0].keys() for v in values)
        and not all(_is_scalar(x) for x in values[0].values())
        and not _record_groups(values[0])
    )


def _scalar_table(d: dict) -> list[str]:
    items = list(d.items())
    numbers = [v for _, v in items if isinstance(v, (int, float)) and not isinstance(v, bool)]
    distribution = len(numbers) == len(items) > 1 and all(isinstance(v, float) and 0 <= v <= 1 for v in numbers) and abs(sum(numbers) - 1) < 0.001
    if len(items) > MAX_ROWS and all(isinstance(v, (int, float)) for _, v in items):
        items.sort(key=lambda kv: -kv[1])
    return _table(["", "value"], [[_num(k), _num(v, distribution or _is_share_key(str(k)))] for k, v in items])


PAIR_SEPARATORS = (" -> ", " / ")


def _pair_separator(d: dict) -> str | None:
    if len(d) < 4 or not all(isinstance(v, (int, float)) for v in d.values()):
        return None
    for sep in PAIR_SEPARATORS:
        if all(str(k).count(sep) == 1 for k in d):
            return sep
    return None


def _pair_matrix(d: dict, sep: str) -> list[str]:
    """Counts keyed by ``row<sep>column`` as a matrix of the most frequent rows and columns."""
    cells = {tuple(str(k).split(sep)): v for k, v in d.items()}
    row_totals: dict[str, float] = {}
    col_totals: dict[str, float] = {}
    for (r, c), v in cells.items():
        row_totals[r] = row_totals.get(r, 0) + v
        col_totals[c] = col_totals.get(c, 0) + v
    rows = sorted(row_totals, key=row_totals.get, reverse=True)[:MAX_COLUMNS]
    columns = sorted(col_totals, key=col_totals.get, reverse=True)[:MAX_COLUMNS]
    body = [[r, *(_num(cells.get((r, c), 0)) for c in columns)] for r in rows]
    lines = _table([sep.strip(), *columns], body)
    hidden = len(cells) - sum(1 for r, c in cells if r in rows and c in columns)
    if hidden:
        lines.insert(-1, f"\n*Rows and columns are the {MAX_COLUMNS} most frequent values; {hidden} rarer pairs not shown.*")
    return lines


def _numeric_matrix(d: dict) -> bool:
    if not _uniform(d, lambda v: isinstance(v, dict) and bool(v) and all(isinstance(x, (int, float)) for x in v.values())):
        return False
    return _shared_values({k: {"cells": v} for k, v in d.items()}, "cells")


def _matrix_table(d: dict, label: str) -> list[str]:
    columns: list[str] = []
    for v in d.values():
        columns += [c for c in v if c not in columns]
    flat = all(len(v) == len(columns) for v in d.values()) or len(columns) <= MAX_COLUMNS
    if not flat:
        return []
    percent = all(isinstance(x, float) and 0 <= x <= 1 for v in d.values() for x in v.values()) and (
        "share" in label or all(sum(v.values()) <= 1.001 for v in d.values())
    )
    rows = [[_num(k), *(_num(v.get(c), percent) for c in columns)] for k, v in d.items()]
    return _table([label, *columns], rows)


def _records_table(records: list[dict]) -> list[str]:
    columns: list[str] = []
    for r in records:
        columns += [c for c in r if c not in columns and _is_scalar(r[c])]
    rows = [[_num(r.get(c), _is_share_key(c)) for c in columns] for r in records]
    return _table(columns, rows)


def render_value(key: str, value, depth: int) -> list[str]:
    heading = ("#" * min(depth, 6) + " " if depth <= 6 else "**") + _title(key) + ("" if depth <= 6 else "**")
    lines = [heading, ""]
    if isinstance(value, list):
        if value and all(isinstance(r, dict) for r in value):
            return lines + _records_table(value)
        return lines + [", ".join(_num(v) for v in value), ""]
    if not isinstance(value, dict):
        return lines + [_num(value, _is_share_key(key)), ""]
    if not value:
        return lines + ["*empty*", ""]
    if value.get("available") is False:
        return lines + [f"*Not available:* {value.get('reason', '')}", ""]
    if _is_summary(value):
        return lines + _summary_table({key: value}, "")
    if _is_ci_shares(value):
        return lines + _ci_table(value)
    if _is_counts(value) and all(_is_scalar(v) or k in ("counts", "shares") for k, v in value.items()):
        return lines + _counts_table(value)
    if _uniform(value, _is_summary):
        return lines + _summary_table(value, "group") + _below_min_note(value)
    if _uniform(value, _is_ci_shares) and _shared_values(value, "shares"):
        return lines + _group_matrix(value, "group", "shares")
    plain_counts = _uniform(value, lambda v: _is_counts(v) and all(_is_scalar(x) or k in ("counts", "shares") for k, x in v.items()))
    if plain_counts and _shared_values(value, "counts"):
        cells = "shares" if all("shares" in v for v in value.values() if not _is_below_min(v)) else "counts"
        return lines + _group_matrix(value, "group", cells)
    if _uniform(value, _is_ci):
        return lines + _ci_groups_table(value, "group")
    if _uniform(value, lambda v: isinstance(v, dict) and bool(v) and all(_is_ci(s) for s in v.values())):
        wrapped = {k: v if _is_below_min(v) else {"shares": v} for k, v in value.items()}
        if _shared_values(wrapped, "shares"):
            return lines + _group_matrix(wrapped, "group", "shares")
    if not plain_counts and _record_groups(value):
        return lines + _record_groups_table(value, "group")
    sep = _pair_separator(value)
    if sep:
        return lines + _pair_matrix(value, sep)
    if all(_is_scalar(v) for v in value.values()):
        return lines + _scalar_table(value)
    if _numeric_matrix(value):
        table = _matrix_table(value, key)
        if table:
            return lines + table

    if _pivot(value):
        fields = list(next(iter(value.values())))
        flat = [f for f in fields if all(_is_scalar(v[f]) for v in value.values())]
        if flat:
            rows = [[_num(g), *(_num(v[f], _is_share_key(f)) for f in flat)] for g, v in value.items()]
            lines += _table(["group", *flat], rows)
        for field in fields:
            if field not in flat:
                lines += render_value(field, {g: v[field] for g, v in value.items()}, depth + 1)
        return lines

    scalars = {k: v for k, v in value.items() if _is_scalar(v)}
    if scalars:
        lines += _scalar_table(scalars)
    for k, v in value.items():
        if k not in scalars:
            lines += render_value(str(k), v, depth + 1)
    return lines


def render(report: dict, source: Path) -> str:
    lines = [
        f"# Dataset analysis: {report.get('group', source.stem)}",
        "",
        f"Source `{source.name}`, generated {report.get('generated_at', '?')}, "
        f"{_num(report.get('episodes'))} episodes.",
        "",
    ]
    for key, value in report.get("entries", {}).items():
        lines += render_value(key, value, 2)
    return "\n".join(lines).rstrip() + "\n"


def render_file(path: Path, out: Path | None = None) -> Path:
    out = out or path.with_suffix(".md")
    out.write_text(render(json.loads(path.read_text()), path))
    return out
