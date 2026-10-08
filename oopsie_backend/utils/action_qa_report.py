"""Render action_qa.json (from run_action_qa.py) as a standalone HTML page: a dataset x
check grid of flagged/checked episode counts, with each cell linking to the flagged
episodes and the values that tripped the check.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path

# Shown as the column-header tooltip; mirrors the one-line comments in action_qa_config.yaml.
CHECK_DESCRIPTIONS = {
    ("general", "duplicate_pairs"): "Two different action types hold the same values (a likely copy-paste bug)",
    ("general", "length_mismatch"): "An action and its matching robot state have different numbers of steps",
    ("general", "scale_consistency"): "Actions are on a very different scale from the robot state",
    ("general", "tracking_residual"): "The commanded position doesn't match where the robot actually ends up",
    ("general", "delta_detection"): "Actions were saved as step-to-step changes instead of absolute positions",
    ("cartesian_position", "quaternion_shape"): "Pose isn't [x, y, z, qx, qy, qz, qw] per arm with a unit-length quaternion",
    ("cartesian_position", "position_units"): "Positions aren't in metres (cm/mm or a far-away frame)",
    ("cartesian_position", "quaternion_sign_flip"): "The quaternion suddenly flips sign (q -> -q)",
    ("cartesian_position", "rotation_jump"): "The commanded orientation turns unrealistically fast between two steps",
    ("cartesian_position", "rotation_tracking"): "The commanded orientation doesn't match where the robot's orientation ends up",
    ("cartesian_position", "delta_detection"): "xyz actions were saved as step-to-step changes instead of absolute positions",
    ("cartesian_position", "biarm_ordering"): "Left and right arms look swapped in the actions",
}

MAX_ROWS_PER_CHECK = 50

_STYLE = """
:root {
  --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; --ink: #0f172a; --muted: #64748b;
  --ok: #dcfce7; --ok-ink: #166534; --na: #f1f5f9; --bad-rgb: 220, 38, 38; --link: #0369a1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b1120; --surface: #111827; --border: #1f2937; --ink: #e5e7eb; --muted: #94a3b8;
    --ok: #14532d; --ok-ink: #bbf7d0; --na: #1e293b; --bad-rgb: 239, 68, 68; --link: #7dd3fc;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
h1 { margin: 0 0 .25rem; font-size: 1.35rem; }
h2 { margin: 2rem 0 .75rem; font-size: 1.1rem; }
h3 { margin: 1.25rem 0 .4rem; font-size: 1rem; }
.muted { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: .75rem 0; color: var(--muted); }
.sw { display: inline-block; width: .9rem; height: .9rem; border-radius: 3px; vertical-align: -2px; margin-right: .3rem;
  border: 1px solid var(--border); }
.scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; }
th, td { padding: .35rem .5rem; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
.grid th { font-weight: 600; font-size: .8rem; white-space: nowrap; background: var(--surface); }
.grid thead th.cat { text-align: center; border-left: 2px solid var(--border); }
.grid thead tr + tr th { white-space: normal; max-width: 6.5rem; vertical-align: bottom; }
.grid th.first, .grid td.first { border-left: 2px solid var(--border); }
.grid td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.grid td.cell { text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 3.5rem; }
.grid td.cell a { color: inherit; text-decoration: none; display: block; }
.grid td.ok { background: var(--ok); color: var(--ok-ink); }
.grid td.na { background: var(--na); color: var(--muted); }
.grid td.bad a { font-weight: 600; }
.grid tbody tr:hover td.name { text-decoration: underline; }
details { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: .4rem 0; }
details > summary { cursor: pointer; padding: .45rem .7rem; }
details[open] > summary { border-bottom: 1px solid var(--border); }
details table { font-size: .82rem; }
details td.ep { white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
details td.det { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-word; }
:target { outline: 2px solid var(--link); }
a { color: var(--link); }
pre { margin: 0; padding: .7rem; overflow-x: auto; font-size: .8rem; }
"""

# Opens the <details> a grid cell links to.
_SCRIPT = """
function openTarget() {
  const el = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (el && el.tagName === "DETAILS") { el.open = true; el.scrollIntoView({block: "start"}); }
}
window.addEventListener("hashchange", openTarget);
openTarget();
"""


def _slug(*parts: str) -> str:
    return "--".join(re.sub(r"[^A-Za-z0-9]+", "-", p).strip("-") for p in parts)


def _short(dataset_repo: str) -> str:
    return dataset_repo.rsplit("/", 1)[-1]


def _is_arm_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) and "arm" in v for v in value)


def _fmt(value) -> str:
    """Compact one-line rendering of a finding's evidence values."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        if "arm" in value:  # one arm's entry: "arm 1: count=3, ..."
            rest = {k: v for k, v in value.items() if k != "arm"}
            return f"arm {value['arm']}: {_fmt(rest)}"
        # Per-arm lists read fine without their "arms=" label.
        return ", ".join(_fmt(v) if k == "arms" and _is_arm_list(v) else f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            return "; ".join(_fmt(v) for v in value)
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _checks(report: dict) -> list[tuple[str, str]]:
    """(category, test) columns in config order, from the first dataset's entry."""
    for stats in report.get("per_dataset", {}).values():
        return [(cat, test) for cat, tests in stats["tests"].items() for test in tests]
    return []


def _cell(dataset_repo: str, category: str, test: str, stats: dict, *, first: bool) -> str:
    checked = stats.get("checked_episodes", 0)
    flagged = stats.get("flagged_episodes", 0)
    skips = stats.get("skips") or {}
    tip = [f"{flagged} of {checked} checked episodes flagged"]
    if skips:
        tip.append("skipped: " + ", ".join(f"{reason} x{n}" for reason, n in sorted(skips.items())))
    title = escape(" | ".join(tip), quote=True)
    first = " first" if first else ""  # left border separating categories

    if checked == 0:
        return f'<td class="cell na{first}" title="{title}">{"skipped" if skips else "–"}</td>'
    mark = "*" if skips else ""
    if flagged == 0:
        return f'<td class="cell ok{first}" title="{title}">0/{checked}{mark}</td>'
    share = flagged / checked
    alpha = 0.15 + 0.85 * share
    ink = "#fff" if alpha > 0.55 else "inherit"
    href = "#" + _slug(dataset_repo, category, test)
    return (
        f'<td class="cell bad{first}" title="{title}" style="background: rgba(var(--bad-rgb), {alpha:.2f}); color: {ink}">'
        f'<a href="{href}">{flagged}/{checked}{mark}</a></td>'
    )


def _grid(report: dict, checks: list[tuple[str, str]]) -> str:
    categories: dict[str, list[str]] = defaultdict(list)
    for cat, test in checks:
        categories[cat].append(test)

    head_top = ['<th rowspan="2">Dataset</th>', '<th rowspan="2">Episodes</th>', '<th rowspan="2">Flagged</th>']
    head_bottom = []
    for cat, tests in categories.items():
        head_top.append(f'<th class="cat" colspan="{len(tests)}">{escape(cat)}</th>')
        for i, test in enumerate(tests):
            desc = escape(CHECK_DESCRIPTIONS.get((cat, test), ""), quote=True)
            cls = ' class="first"' if i == 0 else ""
            head_bottom.append(f'<th{cls} title="{desc}">{escape(test.replace("_", " "))}</th>')

    def sort_key(item):
        repo, stats = item
        n = stats["episodes_checked"] or 1
        return (-stats["episodes_flagged"] / n, _short(repo).lower())

    rows = []
    for repo, stats in sorted(report["per_dataset"].items(), key=sort_key):
        n, flagged = stats["episodes_checked"], stats["episodes_flagged"]
        pct = f"{100 * flagged / n:.0f}%" if n else "–"
        cells = []
        for cat, tests in categories.items():
            for i, test in enumerate(tests):
                cells.append(_cell(repo, cat, test, stats["tests"][cat][test], first=i == 0))
        errors = f' <span class="muted" title="episodes that failed to open">({stats["errors"]} err)</span>' if stats["errors"] else ""
        rows.append(
            f'<tr><td class="name"><a href="#{_slug(repo)}">{escape(_short(repo))}</a></td>'
            f'<td class="num">{n}{errors}</td><td class="num">{flagged} ({pct})</td>{"".join(cells)}</tr>'
        )
    return (
        '<div class="scroll"><table class="grid"><thead>'
        f'<tr>{"".join(head_top)}</tr><tr>{"".join(head_bottom)}</tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _drilldown(report: dict, checks: list[tuple[str, str]]) -> str:
    by_dataset: dict[str, dict[tuple[str, str], list[tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for episode in report.get("flagged_episodes", []):
        for finding in episode["findings"]:
            by_dataset[episode["dataset_repo"]][(finding["category"], finding["test"])].append(
                (episode["repo_path"], finding)
            )

    sections = []
    for repo in sorted(by_dataset, key=lambda r: _short(r).lower()):
        stats = report["per_dataset"].get(repo, {}).get("tests", {})
        blocks = []
        for cat, test in checks:
            hits = by_dataset[repo].get((cat, test))
            if not hits:
                continue
            checked = stats.get(cat, {}).get(test, {}).get("checked_episodes", "?")
            episodes = len({path for path, _ in hits})
            rows = []
            for path, finding in hits[:MAX_ROWS_PER_CHECK]:
                details = {k: v for k, v in finding.items() if k not in ("category", "test", "key")}
                rows.append(
                    f'<tr><td class="ep">{escape(path)}</td><td>{escape(finding["key"])}</td>'
                    f'<td class="det">{escape(_fmt(details))}</td></tr>'
                )
            more = (
                f'<tr><td colspan="3" class="muted">+{len(hits) - MAX_ROWS_PER_CHECK} more in action_qa.json</td></tr>'
                if len(hits) > MAX_ROWS_PER_CHECK
                else ""
            )
            blocks.append(
                f'<details id="{_slug(repo, cat, test)}"><summary><b>{escape(cat)} · {escape(test)}</b> '
                f'<span class="muted">— {episodes} of {checked} episodes</span></summary>'
                f'<table><tr><th>Episode</th><th>Key</th><th>Details</th></tr>{"".join(rows)}{more}</table></details>'
            )
        sections.append(f'<h3 id="{_slug(repo)}">{escape(_short(repo))}</h3>{"".join(blocks)}')
    return "".join(sections) or '<p class="muted">Nothing flagged.</p>'


def render_action_qa_report(report: dict, output_path: Path) -> None:
    checks = _checks(report)
    totals = report.get("totals", {})
    generated = report.get("generated_at", {})
    if isinstance(generated, dict):
        generated = generated.get("cdt") or generated.get("utc") or ""

    listed = len(report.get("flagged_episodes", []))
    truncated = (
        f'<p class="muted">Only the first {listed} of {totals.get("episodes_flagged", 0)} flagged episodes are '
        "listed below (the grid counts are complete).</p>"
        if listed < totals.get("episodes_flagged", 0)
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action QA</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
<h1>Action QA</h1>
<p class="muted">Generated {escape(str(generated))} · {escape(str(report.get("scope", "")))} ·
{totals.get("episodes_checked", 0)} episodes checked · {totals.get("episodes_flagged", 0)} flagged ·
{totals.get("episodes_skipped", 0)} errored</p>
<div class="legend">
<span>Each cell: <b>flagged / checked</b> episodes. Hover a column for what it checks, a cell for skip reasons; click a red cell for the episodes.</span>
<span><span class="sw" style="background: var(--ok)"></span>none flagged</span>
<span><span class="sw" style="background: rgba(var(--bad-rgb), .3)"></span><span class="sw" style="background: rgba(var(--bad-rgb), 1)"></span>more flagged = darker</span>
<span><span class="sw" style="background: var(--na)"></span>not applicable / skipped</span>
<span>* some episodes skipped</span>
</div>
{_grid(report, checks)}
<h2>Flagged episodes</h2>
{truncated}
{_drilldown(report, checks)}
<h2>Config used</h2>
<details><summary>Thresholds and toggles for this run</summary><pre>{escape(json.dumps(report.get("config_snapshot", {}), indent=2))}</pre></details>
</div>
<script>{_SCRIPT}</script>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
