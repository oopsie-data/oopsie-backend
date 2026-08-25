r"""Serve a lightweight local web UI for labeling face presence in videos.

Example:
    uv run python -m oopsie_backend.scripts.label_face_blur_videos \
        --manifest /datastor1/jiahuikchen/face_blur_testset/manifest.csv

Then open the printed localhost URL and use:
    y = face present, n = no face, u = uncertain, arrows = previous/next
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ALLOWED_LABELS = {"", "yes", "no", "uncertain"}
ALLOWED_BLUR_ACTIONS = {"", "blur", "preserve", "review"}
ALLOWED_EXEMPTION_REASONS = {
    "",
    "already_blurred",
    "task_relevant_depiction",
    "other",
}
REQUIRED_COLUMNS = ("sample_id", "local_video_path", "face_present", "labeler", "notes")
ADVANCED_COLUMNS = (
    "blur_action",
    "exemption_reason",
    "annotation_source",
    "annotated_at",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face-Blur Video Labeler</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --yes: #0f8a4b;
      --no: #b42318;
      --uncertain: #9a6700;
      --blur: #b42318;
      --preserve: #175cd3;
      --review: #9a6700;
      --focus: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .summary {
      display: flex;
      gap: 12px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      padding: 16px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .viewer, .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .viewer {
      padding: 12px;
    }
    video {
      display: block;
      width: 100%;
      max-height: calc(100vh - 250px);
      background: #111827;
      border-radius: 6px;
      object-fit: contain;
    }
    .meta {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 6px 10px;
      padding: 12px 2px 0;
      color: var(--muted);
      font-size: 13px;
    }
    .meta div:nth-child(even) {
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: #98a2b3; }
    button.primary { background: var(--focus); color: #fff; border-color: var(--focus); }
    button.yes { border-color: var(--yes); color: var(--yes); }
    button.no { border-color: var(--no); color: var(--no); }
    button.uncertain { border-color: var(--uncertain); color: var(--uncertain); }
    button.blur { border-color: var(--blur); color: var(--blur); }
    button.preserve { border-color: var(--preserve); color: var(--preserve); }
    button.review { border-color: var(--review); color: var(--review); }
    button.selected.yes { background: #e7f6ee; }
    button.selected.no { background: #fff0ed; }
    button.selected.uncertain { background: #fff7db; }
    button.selected.blur { background: #fff0ed; }
    button.selected.preserve { background: #eff8ff; }
    button.selected.review { background: #fff7db; }
    .decision-help {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .form-row {
      display: grid;
      gap: 6px;
      margin-top: 12px;
    }
    label {
      color: var(--muted);
      font-size: 13px;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 72px;
      resize: vertical;
    }
    .status {
      min-height: 22px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .side {
      display: flex;
      flex-direction: column;
      max-height: calc(100vh - 88px);
      overflow: hidden;
    }
    .side-head {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .list {
      overflow: auto;
      padding: 6px;
    }
    .row {
      display: grid;
      grid-template-columns: 54px 1fr auto;
      gap: 8px;
      align-items: center;
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 7px 8px;
      background: transparent;
    }
    .row:hover, .row.active {
      border-color: var(--line);
      background: #f8fafc;
    }
    .row-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge {
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: #475467;
      background: #eef2f6;
      text-transform: capitalize;
    }
    .badge.yes { color: var(--yes); background: #e7f6ee; }
    .badge.no { color: var(--no); background: #fff0ed; }
    .badge.uncertain { color: var(--uncertain); background: #fff7db; }
    .badge.blur { color: var(--blur); background: #fff0ed; }
    .badge.preserve { color: var(--preserve); background: #eff8ff; }
    .badge.review, .badge.needs-action { color: var(--review); background: #fff7db; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .side { max-height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Face-Blur Video Labeler</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section class="viewer">
      <video id="video" controls playsinline autoplay muted loop></video>
      <div class="meta">
        <div>Sample</div><div id="sampleId"></div>
        <div>Repository</div><div id="sourceRepo"></div>
        <div>Source</div><div id="sourcePath"></div>
        <div>Video</div><div id="videoPath"></div>
        <div>Queue reason</div><div id="candidateReason"></div>
        <div>Model signals</div><div id="modelSignals"></div>
        <div>Imported label</div><div id="labelSource"></div>
        <div>Current decision</div><div id="currentDecision"></div>
      </div>
      <div class="controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="blurBtn" class="blur" type="button">B · Face → Blur</button>
        <button id="noBtn" class="no" type="button">N · No Face</button>
        <button id="alreadyBlurredBtn" class="preserve advanced-only" type="button">A · Already Blurred</button>
        <button id="taskRelevantBtn" class="preserve advanced-only" type="button">P · Task/Photo → Preserve</button>
        <button id="otherPreserveBtn" class="preserve advanced-only" type="button">O · Other → Preserve</button>
        <button id="uncertainBtn" class="review" type="button">U · Uncertain/Review</button>
        <button id="clearBtn" type="button">Clear</button>
        <button id="nextBtn" type="button">Next</button>
      </div>
      <div class="decision-help advanced-only">
        Face presence and blur policy are stored separately. “Already blurred” and
        “Task/Photo” both record that a face exists while instructing the downstream
        pipeline to preserve the video.
      </div>
      <div class="form-row">
        <label for="labeler">Labeler</label>
        <input id="labeler" autocomplete="name">
      </div>
      <div class="form-row">
        <label for="notes">Notes</label>
        <textarea id="notes" placeholder="Optional edge cases, timestamps, reflections, partial faces..."></textarea>
      </div>
      <div class="status" id="status"></div>
    </section>
    <aside class="side">
      <div class="side-head">
        <select id="filter">
          <option value="all">All videos</option>
          <option value="incomplete" selected>Needs annotation/action</option>
          <option value="blur">Blur</option>
          <option value="preserve">Preserve</option>
          <option value="yes">Face present</option>
          <option value="no">No face</option>
          <option value="review">Uncertain/review</option>
          <option value="imported">Imported labels</option>
        </select>
        <input id="search" placeholder="Search sample/source">
      </div>
      <div class="list" id="list"></div>
    </aside>
  </main>

  <script>
    let rows = [];
    let current = 0;
    let saveTimer = null;
    let advancedMode = false;
    const els = {
      summary: document.getElementById('summary'),
      video: document.getElementById('video'),
      sampleId: document.getElementById('sampleId'),
      sourceRepo: document.getElementById('sourceRepo'),
      sourcePath: document.getElementById('sourcePath'),
      videoPath: document.getElementById('videoPath'),
      candidateReason: document.getElementById('candidateReason'),
      modelSignals: document.getElementById('modelSignals'),
      labelSource: document.getElementById('labelSource'),
      currentDecision: document.getElementById('currentDecision'),
      status: document.getElementById('status'),
      labeler: document.getElementById('labeler'),
      notes: document.getElementById('notes'),
      filter: document.getElementById('filter'),
      search: document.getElementById('search'),
      list: document.getElementById('list'),
      blurBtn: document.getElementById('blurBtn'),
      noBtn: document.getElementById('noBtn'),
      alreadyBlurredBtn: document.getElementById('alreadyBlurredBtn'),
      taskRelevantBtn: document.getElementById('taskRelevantBtn'),
      otherPreserveBtn: document.getElementById('otherPreserveBtn'),
      uncertainBtn: document.getElementById('uncertainBtn'),
    };

    async function load() {
      const res = await fetch('/api/manifest');
      const data = await res.json();
      rows = data.rows;
      advancedMode = Boolean(data.advanced);
      for (const element of document.querySelectorAll('.advanced-only')) {
        element.hidden = !advancedMode;
      }
      current = firstUnlabeledIndex();
      render();
    }

    function firstUnlabeledIndex() {
      const idx = rows.findIndex(row => !decisionComplete(row));
      return idx >= 0 ? idx : 0;
    }

    function decisionComplete(row) {
      if (!advancedMode) return ['yes', 'no', 'uncertain'].includes(row.face_present);
      if (row.face_present === 'no') return row.blur_action === 'preserve';
      if (row.face_present === 'yes' && row.blur_action === 'blur') return true;
      if (row.face_present === 'yes' && row.blur_action === 'preserve') {
        return ['already_blurred', 'task_relevant_depiction', 'other'].includes(row.exemption_reason);
      }
      return row.face_present === 'uncertain' && row.blur_action === 'review';
    }

    function decisionBadge(row) {
      if (!decisionComplete(row)) {
        return row.face_present ? ['needs action', 'needs-action'] : ['unlabeled', ''];
      }
      if (!advancedMode) {
        if (row.face_present === 'yes') return ['face', 'yes'];
        if (row.face_present === 'no') return ['no face', 'no'];
        return ['uncertain', 'uncertain'];
      }
      if (row.face_present === 'no') return ['no face', 'no'];
      if (row.blur_action === 'blur') return ['blur', 'blur'];
      if (row.blur_action === 'review') return ['review', 'review'];
      const reasons = {
        already_blurred: 'preserve · blurred',
        task_relevant_depiction: 'preserve · task/photo',
        other: 'preserve · other',
      };
      return [reasons[row.exemption_reason] || 'preserve', 'preserve'];
    }

    function filteredRows() {
      const filter = els.filter.value;
      const q = els.search.value.trim().toLowerCase();
      return rows.map((row, index) => ({row, index})).filter(({row}) => {
        const label = row.face_present || '';
        if (filter === 'incomplete' && decisionComplete(row)) return false;
        if (['yes', 'no'].includes(filter) && label !== filter) return false;
        if (['blur', 'preserve'].includes(filter) && row.blur_action !== filter) return false;
        if (filter === 'review' && row.blur_action !== 'review') return false;
        if (filter === 'imported' && !(row.annotation_source || '').startsWith('imported_')) return false;
        if (q) {
          const hay = [row.sample_id, row.source_repo, row.source_path,
            row.local_video_path, row.notes, row.candidate_reason,
            row.annotation_source].join(' ').toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });
    }

    function render() {
      if (!rows.length) return;
      const row = rows[current];
      els.video.src = `/media/${encodeURIComponent(row.sample_id)}`;
      els.video.play().catch(() => {});
      els.sampleId.textContent = `${row.sample_id} (${current + 1}/${rows.length})`;
      els.sourceRepo.textContent = row.source_repo || '';
      els.sourcePath.textContent = row.source_path || '';
      els.videoPath.textContent = row.local_video_path || '';
      els.candidateReason.textContent = row.candidate_reason || '';
      els.modelSignals.textContent = [
        row.qwen_verified_face ? `Qwen=${row.qwen_verified_face}` : '',
        row.deface_positive_frames ? `deface frames=${row.deface_positive_frames}` : '',
        row.qwen_positive_frames ? `Qwen YES frames=${row.qwen_positive_frames}` : '',
        row.disagreement_rank_repo ? `disagreement rank=${row.disagreement_rank_repo}` : '',
      ].filter(Boolean).join(' · ');
      els.labelSource.textContent = [row.annotation_source, row.annotated_at].filter(Boolean).join(' · ');
      els.currentDecision.textContent = [
        row.face_present ? `face=${row.face_present}` : 'face=unlabeled',
        row.blur_action ? `action=${row.blur_action}` : 'action=missing',
        row.exemption_reason ? `reason=${row.exemption_reason}` : '',
      ].filter(Boolean).join(' · ');
      els.labeler.value = row.labeler || localStorage.getItem('faceBlurLabeler') || '';
      els.notes.value = row.notes || '';
      setSelected(row);
      renderSummary();
      renderList();
    }

    function renderSummary() {
      const counts = {blur: 0, preserve: 0, review: 0, incomplete: 0};
      for (const row of rows) {
        if (!decisionComplete(row)) counts.incomplete += 1;
        else if (row.blur_action === 'blur' || (!advancedMode && row.face_present === 'yes')) counts.blur += 1;
        else if (row.blur_action === 'preserve' || (!advancedMode && row.face_present === 'no')) counts.preserve += 1;
        else if (row.blur_action === 'review' || (!advancedMode && row.face_present === 'uncertain')) counts.review += 1;
      }
      els.summary.textContent = `${rows.length} total · ${counts.blur} blur · ${counts.preserve} preserve · ${counts.review} review · ${counts.incomplete} remaining`;
    }

    function renderList() {
      els.list.textContent = '';
      for (const {row, index} of filteredRows()) {
        const btn = document.createElement('button');
        btn.className = `row${index === current ? ' active' : ''}`;
        btn.type = 'button';
        btn.addEventListener('click', () => {
          current = index;
          render();
        });
        const num = document.createElement('span');
        num.textContent = String(index + 1).padStart(2, '0');
        const name = document.createElement('span');
        name.className = 'row-name';
        name.textContent = row.sample_id;
        name.title = row.source_path || row.local_video_path || '';
        const badge = document.createElement('span');
        const [label, badgeClass] = decisionBadge(row);
        badge.className = `badge ${badgeClass}`;
        badge.textContent = label;
        btn.append(num, name, badge);
        els.list.append(btn);
      }
    }

    function setSelected(row) {
      els.blurBtn.classList.toggle('selected', row.face_present === 'yes' && (!advancedMode || row.blur_action === 'blur'));
      els.noBtn.classList.toggle('selected', row.face_present === 'no');
      els.alreadyBlurredBtn.classList.toggle('selected', row.exemption_reason === 'already_blurred');
      els.taskRelevantBtn.classList.toggle('selected', row.exemption_reason === 'task_relevant_depiction');
      els.otherPreserveBtn.classList.toggle('selected', row.exemption_reason === 'other');
      els.uncertainBtn.classList.toggle('selected', row.face_present === 'uncertain');
    }

    async function save(decision, advance = false) {
      const row = rows[current];
      const labeler = els.labeler.value.trim();
      if (labeler) localStorage.setItem('faceBlurLabeler', labeler);
      const payload = {
        sample_id: row.sample_id,
        face_present: decision.face_present,
        blur_action: advancedMode ? decision.blur_action : '',
        exemption_reason: advancedMode ? decision.exemption_reason : '',
        labeler,
        notes: els.notes.value,
      };
      els.status.textContent = 'Saving...';
      const res = await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        els.status.textContent = `Save failed: ${await res.text()}`;
        return;
      }
      const data = await res.json();
      rows = data.rows;
      els.status.textContent = 'Saved';
      setSelected(rows[current]);
      renderSummary();
      renderList();
      if (advance) next();
    }

    function queueNotesSave() {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => save({
        face_present: rows[current].face_present || '',
        blur_action: rows[current].blur_action || '',
        exemption_reason: rows[current].exemption_reason || '',
      }, false), 450);
    }

    function next() {
      const nextIndex = filteredRows().map(item => item.index).find(index => index > current);
      if (nextIndex !== undefined) {
        current = nextIndex;
        render();
      }
    }

    function previous() {
      const previousIndices = filteredRows().map(item => item.index).filter(index => index < current);
      if (previousIndices.length) {
        current = previousIndices[previousIndices.length - 1];
        render();
      }
    }

    els.blurBtn.addEventListener('click', () => save({face_present: 'yes', blur_action: 'blur', exemption_reason: ''}, true));
    els.noBtn.addEventListener('click', () => save({face_present: 'no', blur_action: 'preserve', exemption_reason: ''}, true));
    els.alreadyBlurredBtn.addEventListener('click', () => save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'already_blurred'}, true));
    els.taskRelevantBtn.addEventListener('click', () => save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'task_relevant_depiction'}, true));
    els.otherPreserveBtn.addEventListener('click', () => save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'other'}, true));
    els.uncertainBtn.addEventListener('click', () => save({face_present: 'uncertain', blur_action: 'review', exemption_reason: ''}, true));
    document.getElementById('clearBtn').addEventListener('click', () => save({face_present: '', blur_action: '', exemption_reason: ''}, false));
    document.getElementById('nextBtn').addEventListener('click', next);
    document.getElementById('prevBtn').addEventListener('click', previous);
    els.notes.addEventListener('input', queueNotesSave);
    els.labeler.addEventListener('change', () => save({
      face_present: rows[current].face_present || '',
      blur_action: rows[current].blur_action || '',
      exemption_reason: rows[current].exemption_reason || '',
    }, false));
    els.filter.addEventListener('change', renderList);
    els.search.addEventListener('input', renderList);

    document.addEventListener('keydown', event => {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (['b', 'B', 'y', 'Y'].includes(event.key)) save({face_present: 'yes', blur_action: 'blur', exemption_reason: ''}, true);
      else if (event.key === 'n' || event.key === 'N') save({face_present: 'no', blur_action: 'preserve', exemption_reason: ''}, true);
      else if (event.key === 'a' || event.key === 'A') save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'already_blurred'}, true);
      else if (event.key === 'p' || event.key === 'P') save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'task_relevant_depiction'}, true);
      else if (event.key === 'o' || event.key === 'O') save({face_present: 'yes', blur_action: 'preserve', exemption_reason: 'other'}, true);
      else if (event.key === 'u' || event.key === 'U') save({face_present: 'uncertain', blur_action: 'review', exemption_reason: ''}, true);
      else if (event.key === 'ArrowRight') next();
      else if (event.key === 'ArrowLeft') previous();
      else if (event.key === ' ') {
        event.preventDefault();
        if (els.video.paused) els.video.play();
        else els.video.pause();
      }
    });

    load().catch(err => {
      els.status.textContent = String(err);
    });
  </script>
</body>
</html>
"""


class ManifestStore:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.base_dir = self.manifest_path.parent
        self.lock = threading.RLock()
        self.fieldnames: list[str] = []
        self.rows: list[dict[str, str]] = []
        self.advanced = False
        self.history_path = self.manifest_path.with_name(
            f"{self.manifest_path.stem}.history.jsonl"
        )
        self.load()

    def load(self) -> None:
        with self.lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        with self.manifest_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"manifest has no header: {self.manifest_path}")
            self.fieldnames = list(reader.fieldnames)
            missing = [col for col in REQUIRED_COLUMNS if col not in self.fieldnames]
            if missing:
                raise ValueError(f"manifest is missing required column(s): {', '.join(missing)}")
            advanced_present = [col in self.fieldnames for col in ADVANCED_COLUMNS]
            if any(advanced_present) and not all(advanced_present):
                missing_advanced = [
                    col for col in ADVANCED_COLUMNS if col not in self.fieldnames
                ]
                raise ValueError(
                    "advanced manifest is missing column(s): "
                    + ", ".join(missing_advanced)
                )
            self.advanced = all(advanced_present)
            self.rows = [
                {key: value or "" for key, value in row.items() if key is not None}
                for row in reader
            ]

    def snapshot(self) -> dict:
        with self.lock:
            self._load_locked()
            rows = [dict(row) for row in self.rows]
        return {
            "manifest": str(self.manifest_path),
            "rows": rows,
            "summary": summarize(rows),
            "advanced": self.advanced,
            "history": str(self.history_path) if self.advanced else None,
        }

    def update_label(
        self,
        sample_id: str,
        *,
        face_present: str,
        blur_action: str,
        exemption_reason: str,
        labeler: str,
        notes: str,
    ) -> dict:
        if face_present not in ALLOWED_LABELS:
            raise ValueError(f"face_present must be one of {sorted(ALLOWED_LABELS)!r}")
        if blur_action not in ALLOWED_BLUR_ACTIONS:
            raise ValueError(
                f"blur_action must be one of {sorted(ALLOWED_BLUR_ACTIONS)!r}"
            )
        if exemption_reason not in ALLOWED_EXEMPTION_REASONS:
            raise ValueError(
                "exemption_reason must be one of "
                f"{sorted(ALLOWED_EXEMPTION_REASONS)!r}"
            )
        if self.advanced:
            self._validate_advanced_decision(
                face_present, blur_action, exemption_reason
            )
        with self.lock:
            self._load_locked()
            for row in self.rows:
                if row.get("sample_id") == sample_id:
                    before = dict(row)
                    row["face_present"] = face_present
                    row["labeler"] = labeler
                    row["notes"] = notes
                    if self.advanced:
                        row["blur_action"] = blur_action
                        row["exemption_reason"] = exemption_reason
                        row["annotation_source"] = "manual_ui" if face_present else ""
                        row["annotated_at"] = utc_now() if face_present else ""
                    self._write_locked()
                    if self.advanced:
                        self._append_history_locked(before, row)
                    return self.snapshot()
        raise KeyError(f"sample_id not found: {sample_id}")

    @staticmethod
    def _validate_advanced_decision(
        face_present: str,
        blur_action: str,
        exemption_reason: str,
    ) -> None:
        if not face_present:
            if blur_action or exemption_reason:
                raise ValueError("cleared face label requires cleared blur fields")
            return
        if face_present == "no":
            if blur_action not in {"", "preserve"} or exemption_reason:
                raise ValueError("no-face label must use preserve with no exemption")
            return
        if face_present == "uncertain":
            if blur_action not in {"", "review"} or exemption_reason:
                raise ValueError("uncertain label must use review with no exemption")
            return
        if not blur_action:
            if exemption_reason:
                raise ValueError("face with no blur action cannot have an exemption")
            return
        if blur_action == "blur":
            if exemption_reason:
                raise ValueError("blur action cannot have an exemption")
            return
        if blur_action == "preserve":
            if exemption_reason not in {
                "already_blurred",
                "task_relevant_depiction",
                "other",
            }:
                raise ValueError("preserved face requires an exemption reason")
            return
        raise ValueError("face-present label must use blur or preserve")

    def _append_history_locked(
        self, before: dict[str, str], after: dict[str, str]
    ) -> None:
        event = {
            "timestamp": utc_now(),
            "sample_id": after.get("sample_id"),
            "before": before,
            "after": dict(after),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def video_path_for(self, sample_id: str) -> Path:
        with self.lock:
            for row in self.rows:
                if row.get("sample_id") == sample_id:
                    raw_path = row.get("local_video_path") or ""
                    path = Path(raw_path)
                    if not path.is_absolute():
                        path = self.base_dir / path
                    return path.resolve()
        raise KeyError(f"sample_id not found: {sample_id}")

    def _write_locked(self) -> None:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.manifest_path.name}.",
            suffix=".tmp",
            dir=str(self.base_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp_name, self.manifest_path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "yes": 0,
        "no": 0,
        "uncertain": 0,
        "unlabeled": 0,
        "blur": 0,
        "preserve": 0,
        "review": 0,
    }
    for row in rows:
        label = row.get("face_present") or ""
        if label in ("yes", "no", "uncertain"):
            counts[label] += 1
        else:
            counts["unlabeled"] += 1
        action = row.get("blur_action") or ""
        if action in ("blur", "preserve", "review"):
            counts[action] += 1
    return counts


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        return None
    start_raw, end_raw = match.groups()
    if not start_raw and not end_raw:
        return None
    if start_raw:
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
    else:
        suffix_len = int(end_raw)
        if suffix_len <= 0:
            return None
        start = max(size - suffix_len, 0)
        end = size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def make_handler(store: ManifestStore):
    class FaceBlurLabelerHandler(BaseHTTPRequestHandler):
        server_version = "FaceBlurLabeler/0.1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/manifest":
                self.send_json(store.snapshot())
                return
            if parsed.path.startswith("/media/"):
                sample_id = unquote(parsed.path[len("/media/") :])
                self.serve_video(sample_id)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path != "/api/label":
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                data = store.update_label(
                    str(payload.get("sample_id") or ""),
                    face_present=str(payload.get("face_present") or ""),
                    blur_action=str(payload.get("blur_action") or ""),
                    exemption_reason=str(payload.get("exemption_reason") or ""),
                    labeler=str(payload.get("labeler") or ""),
                    notes=str(payload.get("notes") or ""),
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_json(data)

        def send_json(self, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_bytes(payload, "application/json; charset=utf-8")

        def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def serve_video(self, sample_id: str) -> None:
            try:
                path = store.video_path_for(sample_id)
            except KeyError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"video not found: {path}")
                return

            size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            byte_range = parse_range(self.headers.get("Range", ""), size)

            if byte_range is None:
                start, end = 0, size - 1
                status = HTTPStatus.OK
            else:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, format: str, *args) -> None:
            print(f"{self.client_address[0]} - {format % args}", file=sys.stderr)

    return FaceBlurLabelerHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a face-blur manifest.csv created by sample_face_blur_videos.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the labeler in the default browser.",
    )
    args = parser.parse_args(argv)

    try:
        store = ManifestStore(args.manifest)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{server.server_port}/"
    summary = summarize(store.rows)
    print(f"Manifest: {store.manifest_path}")
    print(
        "Labels:   "
        f"{summary['yes']} yes, {summary['no']} no, "
        f"{summary['uncertain']} uncertain, {summary['unlabeled']} unlabeled"
    )
    print(f"Serving:  {url}")
    print("Keys:     y=yes, n=no, u=uncertain, left/right=previous/next")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
