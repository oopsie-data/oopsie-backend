"""Render the QA report (``qa/report.py``, action and video checks) as a standalone
HTML page: a dataset x check grid of flagged/checked episode counts, with each cell linking to the flagged
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
    ("general", "action_copies_state"): "Actions are exact copies of the recorded state, not real commands",
    ("general", "static_robot"): "The robot arm barely moves over the whole episode",
    ("general", "action_chunk_shape"): "An arm action step holds a policy's whole action chunk instead of the one action executed",
    ("general", "constant_action"): "An arm action never changes over the whole episode (stuck policy or placeholder actions)",
    ("general", "command_rate"): "Each arm action is held for the same N steps: commands arrive at control_freq / N, not the declared control_freq",
    ("general", "required_state_keys"): "A state the action space needs is missing (gripper_position always; joint/cartesian position for joint/cartesian actions)",
    ("general", "placeholder_metadata"): "Operator, policy, robot or gripper name is blank or a template placeholder",
    ("general", "scale_consistency"): "Gripper position actions are on a very different scale from the state",
    ("general", "tracking_residual"): "The commanded gripper position doesn't match where the gripper ends up",
    ("general", "delta_detection"): "Gripper actions were saved as step-to-step changes instead of absolute positions",
    ("general", "gripper_binary_present"): "No gripper_binary action is recorded",
    ("general", "gripper_range"): "Binary gripper actions fall outside [0, 1] (e.g. -1/+1)",
    ("general", "gripper_command_range"): "Gripper position commands go far beyond the range the gripper state covers",
    ("joint_position", "joint_names"): "Action/state joint names are missing, don't match, or are in an odd order",
    ("joint_position", "dof"): "Number of joint values doesn't match the number of joint names",
    ("joint_position", "radians"): "Joint values aren't in radians (degrees or a -100..100 range)",
    ("joint_position", "step_jump"): "A joint moves unrealistically far between two steps",
    ("joint_position", "gripper_in_joints"): "A gripper value is mixed into the joint vector",
    ("joint_position", "normalized_range"): "Joint actions peak at exactly +-1 (normalized, not radians)",
    ("joint_position", "delta_detection"): "Joint actions were saved as step-to-step changes instead of absolute positions",
    ("joint_position", "tracking"): "The commanded joint positions don't match where the joints actually end up",
    ("joint_velocity", "dof"): "Number of joint velocities doesn't match the number of joint names",
    ("joint_velocity", "speed_limit"): "A joint is commanded faster than a real arm can move",
    ("joint_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("joint_velocity", "tracking"): "Velocity x step time doesn't match how far the joints actually moved",
    ("joint_velocity", "looks_like_position"): "The velocity is really a joint position saved under the wrong key",
    ("joint_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("cartesian_velocity", "shape"): "Velocity isn't 6 values per arm (3 linear + 3 angular)",
    ("cartesian_velocity", "speed_limit"): "The end effector is commanded faster than a real arm can move",
    ("cartesian_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("cartesian_velocity", "tracking"): "Velocity x step time doesn't match how the end effector actually moved and turned",
    ("cartesian_velocity", "linear_angular_order"): "Linear and angular halves look swapped",
    ("cartesian_velocity", "looks_like_position"): "The velocity is really a position saved under the wrong key",
    ("cartesian_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("cartesian_position", "quaternion_shape"): "Pose isn't [x, y, z, qx, qy, qz, qw] per arm with a unit-length quaternion",
    ("cartesian_position", "position_units"): "Positions aren't in metres (cm/mm or a far-away frame)",
    ("cartesian_position", "quaternion_sign_flip"): "The quaternion suddenly flips sign (q -> -q)",
    ("cartesian_position", "rotation_jump"): "The commanded orientation turns unrealistically fast between two steps",
    ("cartesian_position", "normalized_range"): "xyz actions peak at exactly +-1 (normalized, not metres)",
    ("cartesian_position", "pose_tracking"): "The commanded position or orientation doesn't match where the robot ends up",
    ("cartesian_position", "delta_detection"): "xyz actions were saved as step-to-step changes instead of absolute positions",
    ("cartesian_position", "biarm_ordering"): "Left and right arms look swapped in the actions",
    ("cartesian_position", "quaternion_scalar_order"): "The quaternion's scalar (w) component looks like it was written first instead of last (rough heuristic, often unreliable)",
    ("video", "resolution"): "A camera video's shorter side is below the minimum resolution",
    ("video", "decoding"): "A camera video is missing, fails to decode, or decodes to the wrong frame count",
    ("video", "brightness"): "A camera video is too dark or too bright",
    ("video", "fps"): "The video's fps header doesn't match control_freq (playback speed; frames may still align)",
    ("video", "duration"): "The video is longer or shorter than the trajectory: frame count doesn't match the number of steps",
    ("video", "camera_duration_mismatch"): "The cameras of one episode have different durations",
    ("video", "corrupted_frames"): "A single frame is unlike both its neighbours (decoding or recording glitch)",
    ("video", "video_jump"): "The image changes suddenly while the robot state changes smoothly (cut, dropped frames, or video out of step)",
    ("video", "choppy"): "The video repeats frames while the robot moves",
    ("video", "frozen"): "The video stops changing for a while as the robot moves",
    ("video", "recording_gap"): "Robot state and video both jump at the same moment: time looks missing from the recording",
    ("video", "state_jump"): "Robot state jumps in one step while the video doesn't: a state glitch, or state and video out of step",
    ("video", "periodic_skips"): "The recording skips time at a regular period (robot state jumps throughout the motion)",
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
.legend label { color: var(--ink); cursor: pointer; }
.legend input[type="checkbox"] { margin-right: .3rem; cursor: pointer; }
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
.grid td.name.failing { background: rgba(var(--bad-rgb), .85); }
.grid td.name.failing a { color: #fff; font-weight: 600; }
details { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: .4rem 0; }
details > summary { cursor: pointer; padding: .45rem .7rem; }
details[open] > summary { border-bottom: 1px solid var(--border); }
details table { font-size: .82rem; }
details td.ep { white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
details td.det { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-word; }
:target { outline: 2px solid var(--link); }
a { color: var(--link); }
pre { margin: 0; padding: .7rem; overflow-x: auto; font-size: .8rem; }
#panel { position: fixed; top: 0; right: 0; bottom: 0; width: min(960px, 100vw); background: var(--surface);
  border-left: 1px solid var(--border); box-shadow: -6px 0 24px rgba(0,0,0,.18); z-index: 20; display: none;
  flex-direction: column; }
#panel.open { display: flex; }
#panel header { display: flex; gap: .6rem; align-items: center; padding: .7rem 1rem; border-bottom: 1px solid var(--border); }
#panel header h3 { margin: 0; flex: 1; font-size: 1rem; }
#panel .body { overflow: auto; padding: .8rem 1rem 2rem; }
#panel button { font: inherit; padding: .25rem .6rem; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); color: var(--ink); cursor: pointer; }
.ep-list { list-style: none; margin: 0 0 1rem; padding: 0; }
.ep-list li { padding: .4rem .5rem; border: 1px solid var(--border); border-radius: 6px; margin: .3rem 0; cursor: pointer; }
.ep-list li:hover, .ep-list li.active { border-color: var(--link); }
.ep-list li .ep { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; }
.ep-list li .det { color: var(--muted); font-size: .8rem; }
.ep-list li.v-real { border-left: 4px solid rgb(var(--bad-rgb)); } .ep-list li.v-fine { border-left: 4px solid #16a34a; }
.ep-list li.v-unsure { border-left: 4px solid #d97706; }
.vids { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: .6rem; margin: .6rem 0; }
.vids figure { margin: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.vids figure.flagged { border: 2px solid rgb(var(--bad-rgb)); }
.vids video { width: 100%; display: block; background: #000; }
.vids figcaption { font-size: .78rem; padding: .2rem .4rem; }
.chart { border: 1px solid var(--border); border-radius: 6px; padding: .4rem; margin: .5rem 0; }
.chart svg { width: 100%; height: 190px; display: block; }
.chart .ttl { font-size: .82rem; font-weight: 600; }
.chart .lg { font-size: .75rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: .8rem; }
.verdict button.on { color: #fff; border-color: transparent; }
.verdict button[data-v=real].on { background: rgb(var(--bad-rgb)); } .verdict button[data-v=fine].on { background: #16a34a; }
.verdict button[data-v=unsure].on { background: #d97706; }
.vals { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; white-space: pre-wrap; }
"""

# Opens the <details> a grid cell links to; show/hide a category's grid columns.
_SCRIPT = """
function openTarget() {
  const el = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (el && el.tagName === "DETAILS") { el.open = true; el.scrollIntoView({block: "start"}); }
}
window.addEventListener("hashchange", openTarget);
openTarget();

// ---- Evidence panel: a red cell lists its flagged episodes; an episode loads its
// evidence (videos, charts, values) from serve_action_qa_report.py.
const CELL_EPISODES = JSON.parse(document.getElementById("cell-episodes").textContent);
const VKEY = "qa_report_verdicts_v1";
let verdicts = {};
try { verdicts = JSON.parse(localStorage.getItem(VKEY) || "{}"); } catch (e) {}
const saveVerdicts = () => { try { localStorage.setItem(VKEY, JSON.stringify(verdicts)); } catch (e) {} };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const panel = document.getElementById("panel"), pBody = panel.querySelector(".body"), pTitle = panel.querySelector("h3");
let current = null;  // {cell: [dataset, category, test], episodes, index}
// Server URLs built from origin + path: a page opened as http://user:pass@host/ would
// otherwise hand its credentials to every request, which browsers refuse.
const serverUrl = rel => new URL(rel, location.origin + location.pathname).href;

document.getElementById("grid").addEventListener("click", e => {
  const a = e.target.closest("a[data-cell]");
  if (!a) return;
  e.preventDefault();
  const [dataset, category, test] = a.dataset.cell.split("|");
  current = {cell: [dataset, category, test], episodes: CELL_EPISODES[a.dataset.cell] || [], index: -1};
  pTitle.textContent = `${dataset.split("/").pop()} · ${category}.${test} · ${current.episodes.length} flagged episode(s)`;
  panel.classList.add("open");
  renderList();
});
panel.querySelector("[data-act=close]").addEventListener("click", () => panel.classList.remove("open"));
document.addEventListener("keydown", e => { if (e.key === "Escape") panel.classList.remove("open"); });
panel.querySelector("[data-act=export]").addEventListener("click", exportVerdicts);

const vkey = (ep) => `${current.cell[0]}|${ep.p}|${current.cell[1]}.${current.cell[2]}`;

function renderList() {
  const items = current.episodes.map((ep, i) => {
    const v = verdicts[vkey(ep)];
    return `<li data-i="${i}" class="${i === current.index ? "active" : ""} ${v ? "v-" + v : ""}">
      <div class="ep">${esc(ep.id || ep.p)}</div><div class="det">${esc(ep.d)}</div></li>`;
  }).join("");
  pBody.innerHTML = `<ul class="ep-list">${items || "<li>No episode details stored for this cell.</li>"}</ul><div id="evidence"></div>`;
  pBody.querySelectorAll(".ep-list li[data-i]").forEach(li => li.addEventListener("click", () => showEvidence(+li.dataset.i)));
}

async function showEvidence(i) {
  current.index = i;
  renderList();
  const ep = current.episodes[i], box = document.getElementById("evidence");
  box.innerHTML = `<p class="muted">Loading evidence for ${esc(ep.id || ep.p)}…</p>`;
  box.scrollIntoView({block: "start"});
  const [dataset, category, test] = current.cell;
  const q = new URLSearchParams({dataset, repo_path: ep.p, category, test});
  let ev;
  try {
    const r = await fetch(serverUrl("evidence?" + q), {credentials: "same-origin"});
    ev = await r.json();
    if (!r.ok) throw new Error(ev.error || r.status);
  } catch (err) {
    box.innerHTML = `<p class="muted">Couldn't load evidence (${esc(err.message)}). Open this report through
      <code>serve_action_qa_report.py</code> to see videos and charts.</p>`;
    return;
  }
  const v = verdicts[vkey(ep)];
  const findings = (ev.findings || []).map(f => esc(f.issue || ep.d)).filter(Boolean);
  const seek = ev.seek_s != null ? Math.max(ev.seek_s - 1, 0) : 0;
  const vids = (ev.videos || []).map(x => `<figure class="${x.flagged ? "flagged" : ""}">
      <video src="${esc(serverUrl(x.url))}" controls muted playsinline preload="metadata" data-seek="${seek}"></video>
      <figcaption>${esc(x.cam)}${x.flagged ? " · flagged" : ""}</figcaption></figure>`).join("");
  box.innerHTML = `
    <h3>${esc(ep.id || ep.p)}</h3>
    <div>${(findings.length ? [...new Set(findings)] : [esc(ep.d)]).join("<br>")}</div>
    <p class="verdict">Verdict:
      <button data-v="real" class="${v === "real" ? "on" : ""}">Real issue</button>
      <button data-v="fine" class="${v === "fine" ? "on" : ""}">False alarm</button>
      <button data-v="unsure" class="${v === "unsure" ? "on" : ""}">Unsure</button></p>
    ${vids ? `<p>${ev.seek_s != null ? `<button data-act="jump">Play all from ${seek.toFixed(1)}s (issue at ${ev.seek_s.toFixed(1)}s)</button>` : ""}
      <button data-act="all">Play all together</button>
      <label><input type="checkbox" data-act="slow"> slow motion (0.25x)</label></p><div class="vids">${vids}</div>` : ""}
    ${(ev.charts || []).map(chartHtml).join("")}
    <details><summary>Values</summary><div class="vals">${esc(JSON.stringify(ev.values, null, 1))}</div></details>`;
  box.querySelectorAll("video").forEach(vid => vid.addEventListener("loadedmetadata", () => { vid.currentTime = +vid.dataset.seek; }, {once: true}));
  box.querySelectorAll(".verdict button").forEach(b => b.addEventListener("click", () => {
    const key = vkey(ep);
    verdicts[key] = verdicts[key] === b.dataset.v ? undefined : b.dataset.v;
    if (!verdicts[key]) delete verdicts[key];
    saveVerdicts();
    box.querySelectorAll(".verdict button").forEach(x => x.classList.toggle("on", x.dataset.v === verdicts[key]));
    const li = pBody.querySelector(`.ep-list li[data-i="${i}"]`);
    if (li) li.className = "active" + (verdicts[key] ? " v-" + verdicts[key] : "");
  }));
  const videos = [...box.querySelectorAll("video")];
  box.querySelector("[data-act=jump]")?.addEventListener("click", () => videos.forEach(x => { x.currentTime = seek; x.play().catch(() => {}); }));
  box.querySelector("[data-act=all]")?.addEventListener("click", () => {
    const t = videos[0]?.currentTime || 0; videos.forEach(x => { x.currentTime = t; x.play().catch(() => {}); });
  });
  box.querySelector("[data-act=slow]")?.addEventListener("change", e => videos.forEach(x => x.playbackRate = e.target.checked ? 0.25 : 1));
}

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"];
function chartHtml(c) {
  const pts = c.series.flatMap(s => s.x.map((x, k) => [x, s.y[k]])).filter(p => p[1] != null);
  if (!pts.length) return `<div class="chart"><div class="ttl">${esc(c.title)}</div><p class="muted">No data</p></div>`;
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = Math.min(...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
  if (c.hline != null) { y0 = Math.min(y0, c.hline); y1 = Math.max(y1, c.hline); }
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const W = 900, H = 190, L = 56, R = 8, T = 8, B = 22;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const lines = c.series.map((s, k) => {
    let d = "", pen = false;
    s.x.forEach((x, j) => { const y = s.y[j]; if (y == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)} `; pen = true; });
    return `<path d="${d}" fill="none" stroke="${COLORS[k % COLORS.length]}" stroke-width="1.6"/>`;
  }).join("");
  const marks = (c.markers || []).filter(m => m >= x0 && m <= x1)
    .map(m => `<line x1="${sx(m)}" x2="${sx(m)}" y1="${T}" y2="${H - B}" stroke="currentColor" stroke-dasharray="4 3"/>`).join("");
  const hl = c.hline != null ? `<line x1="${L}" x2="${W - R}" y1="${sy(c.hline)}" y2="${sy(c.hline)}" stroke="#888" stroke-dasharray="6 4"/>` : "";
  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
  const axes = `<text x="4" y="${T + 10}" font-size="11" fill="currentColor">${fmt(y1)}</text>
    <text x="4" y="${H - B}" font-size="11" fill="currentColor">${fmt(y0)}</text>
    <text x="${L}" y="${H - 4}" font-size="11" fill="currentColor">${x0.toFixed(1)}s</text>
    <text x="${W - R}" y="${H - 4}" font-size="11" fill="currentColor" text-anchor="end">${x1.toFixed(1)}s</text>`;
  const legend = c.series.map((s, k) => `<span><span class="sw" style="background:${COLORS[k % COLORS.length]}"></span>${esc(s.name)}</span>`).join("")
    + ((c.markers || []).length ? `<span>dashed line: flagged moment</span>` : "")
    + (c.hline != null ? `<span>dashed grey: limit</span>` : "");
  return `<div class="chart"><div class="ttl">${esc(c.title)} <span class="muted">(${esc(c.unit || "")})</span></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${hl}${lines}${marks}${axes}</svg><div class="lg">${legend}</div></div>`;
}

function exportVerdicts() {
  const q = s => `"${String(s).replace(/"/g, '""')}"`;
  const rows = [["dataset", "repo_path", "check", "verdict"].join(",")];
  Object.entries(verdicts).forEach(([k, v]) => { const [d, p, c] = k.split("|"); rows.push([d, p, c, v].map(q).join(",")); });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n")], {type: "text/csv"}));
  a.download = "qa_verdicts.csv";
  a.click();
}

function toggleCategory(checkbox) {
  const cells = document.querySelectorAll("#grid .cat-" + checkbox.dataset.cat);
  cells.forEach(el => { el.style.display = checkbox.checked ? "" : "none"; });
}
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
    cls = f"cat-{category}" + (" first" if first else "")  # left border separating categories

    if checked == 0:
        return f'<td class="cell na {cls}" title="{title}">{"skipped" if skips else "–"}</td>'
    mark = "*" if skips else ""
    if flagged == 0:
        return f'<td class="cell ok {cls}" title="{title}">0/{checked}{mark}</td>'
    share = flagged / checked
    alpha = 0.15 + 0.85 * share
    ink = "#fff" if alpha > 0.55 else "inherit"
    href = "#" + _slug(dataset_repo, category, test)
    return (
        f'<td class="cell bad {cls}" title="{title}" style="background: rgba(var(--bad-rgb), {alpha:.2f}); color: {ink}">'
        f'<a href="{href}" data-cell="{escape(dataset_repo, quote=True)}|{category}|{test}">{flagged}/{checked}{mark}</a></td>'
    )


def _category_toggles(categories: dict[str, list[str]]) -> str:
    boxes = "".join(
        f'<label><input type="checkbox" checked data-cat="{escape(cat, quote=True)}" '
        f'onchange="toggleCategory(this)"> {escape(cat)}</label>'
        for cat in categories
    )
    return f'<div class="legend"><span>Show checks:</span>{boxes}</div>'


def _grid(report: dict, checks: list[tuple[str, str]], highlight_ignore: set[str]) -> str:
    categories: dict[str, list[str]] = defaultdict(list)
    for cat, test in checks:
        categories[cat].append(test)

    head_top = ['<th rowspan="2">Dataset</th>', '<th rowspan="2">Episodes</th>', '<th rowspan="2">Flagged</th>']
    head_bottom = []
    for cat, tests in categories.items():
        head_top.append(f'<th class="cat cat-{escape(cat, quote=True)}" colspan="{len(tests)}">{escape(cat)}</th>')
        for i, test in enumerate(tests):
            desc = escape(CHECK_DESCRIPTIONS.get((cat, test), ""), quote=True)
            cls = f"cat-{escape(cat, quote=True)}" + (" first" if i == 0 else "")
            head_bottom.append(f'<th class="{cls}" title="{desc}">{escape(test.replace("_", " "))}</th>')

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
        # Red name = at least one check outside highlight_ignore flagged an episode.
        failing = [
            f"{cat}.{test}"
            for cat, test in checks
            if f"{cat}.{test}" not in highlight_ignore and stats["tests"][cat][test].get("flagged_episodes", 0) > 0
        ]
        name_cls, name_title = ("name failing", "Failing: " + ", ".join(failing)) if failing else ("name", "No counted check flagged")
        rows.append(
            f'<tr><td class="{name_cls}" title="{escape(name_title, quote=True)}"><a href="#{_slug(repo)}">{escape(_short(repo))}</a></td>'
            f'<td class="num">{n}{errors}</td><td class="num">{flagged} ({pct})</td>{"".join(cells)}</tr>'
        )
    return (
        f"{_category_toggles(categories)}"
        '<div class="scroll"><table class="grid" id="grid"><thead>'
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
                f'<tr><td colspan="3" class="muted">+{len(hits) - MAX_ROWS_PER_CHECK} more not listed</td></tr>'
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


def _cell_episodes(report: dict) -> str:
    """``"dataset|category|test"`` -> the flagged episodes of that grid cell, for the
    evidence panel: repo path, episode id and the readable finding details."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for ep in report.get("flagged_episodes", []):
        by_test: dict[tuple[str, str], list[str]] = defaultdict(list)
        for f in ep["findings"]:
            details = {k: v for k, v in f.items() if k not in ("category", "test")}
            by_test[(f["category"], f["test"])].append(_fmt(details))
        for (cat, test), texts in by_test.items():
            cells[f"{ep['dataset_repo']}|{cat}|{test}"].append(
                {"p": ep["repo_path"], "id": str(ep.get("episode_id") or ""), "d": "; ".join(dict.fromkeys(texts))}
            )
    return json.dumps(cells).replace("</", "<\\/")


def action_qa_html(report: dict, highlight_ignore: list[str] | tuple[str, ...] = ()) -> str:
    """``highlight_ignore``: ``category.check`` names that don't turn a dataset's name red."""
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action and video QA</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
<h1>Action and video QA</h1>
<p class="muted">Generated {escape(str(generated))} · {escape(str(report.get("scope", "")))} ·
{totals.get("episodes_checked", 0)} episodes checked · {totals.get("episodes_flagged", 0)} flagged ·
{totals.get("episodes_skipped", 0)} errored</p>
<div class="legend">
<span>Each cell: <b>flagged / checked</b> episodes. Hover a column for what it checks, a cell for skip reasons; click a red cell to see its episodes and their evidence (videos and charts need <code>serve_action_qa_report.py</code>).</span>
<span><span class="sw" style="background: rgba(var(--bad-rgb), .85)"></span>dataset name: at least one check flagged (hover for which; {escape(", ".join(highlight_ignore)) or "no checks"} not counted)</span>
<span><span class="sw" style="background: var(--ok)"></span>none flagged</span>
<span><span class="sw" style="background: rgba(var(--bad-rgb), .3)"></span><span class="sw" style="background: rgba(var(--bad-rgb), 1)"></span>more flagged = darker</span>
<span><span class="sw" style="background: var(--na)"></span>not applicable / skipped</span>
<span>* some episodes skipped</span>
</div>
{_grid(report, checks, set(highlight_ignore))}
<h2>Flagged episodes</h2>
{truncated}
{_drilldown(report, checks)}
<h2>Config used</h2>
<details><summary>Thresholds and toggles for this run</summary><pre>{escape(json.dumps(report.get("config_snapshot", {}), indent=2))}</pre></details>
</div>
<aside id="panel"><header><h3></h3><button data-act="export">Export verdicts (CSV)</button>
<button data-act="close">Close</button></header><div class="body"></div></aside>
<script type="application/json" id="cell-episodes">{_cell_episodes(report)}</script>
<script>{_SCRIPT}</script>
</body>
</html>
"""


def render_action_qa_report(report: dict, output_path: Path, highlight_ignore: list[str] | tuple[str, ...] = ()) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(action_qa_html(report, highlight_ignore), encoding="utf-8")
