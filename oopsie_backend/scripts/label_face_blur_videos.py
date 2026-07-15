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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ALLOWED_LABELS = {"", "yes", "no", "uncertain"}
REQUIRED_COLUMNS = ("sample_id", "local_video_path", "face_present", "labeler", "notes")


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
    button.selected.yes { background: #e7f6ee; }
    button.selected.no { background: #fff0ed; }
    button.selected.uncertain { background: #fff7db; }
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
      <video id="video" controls playsinline></video>
      <div class="meta">
        <div>Sample</div><div id="sampleId"></div>
        <div>Source</div><div id="sourcePath"></div>
        <div>Video</div><div id="videoPath"></div>
      </div>
      <div class="controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="yesBtn" class="yes" type="button">Y · Face</button>
        <button id="noBtn" class="no" type="button">N · No Face</button>
        <button id="uncertainBtn" class="uncertain" type="button">U · Uncertain</button>
        <button id="clearBtn" type="button">Clear</button>
        <button id="nextBtn" type="button">Next</button>
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
          <option value="unlabeled">Unlabeled</option>
          <option value="yes">Face</option>
          <option value="no">No face</option>
          <option value="uncertain">Uncertain</option>
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
    const els = {
      summary: document.getElementById('summary'),
      video: document.getElementById('video'),
      sampleId: document.getElementById('sampleId'),
      sourcePath: document.getElementById('sourcePath'),
      videoPath: document.getElementById('videoPath'),
      status: document.getElementById('status'),
      labeler: document.getElementById('labeler'),
      notes: document.getElementById('notes'),
      filter: document.getElementById('filter'),
      search: document.getElementById('search'),
      list: document.getElementById('list'),
      yesBtn: document.getElementById('yesBtn'),
      noBtn: document.getElementById('noBtn'),
      uncertainBtn: document.getElementById('uncertainBtn'),
    };

    async function load() {
      const res = await fetch('/api/manifest');
      const data = await res.json();
      rows = data.rows;
      current = firstUnlabeledIndex();
      render();
    }

    function firstUnlabeledIndex() {
      const idx = rows.findIndex(row => !row.face_present);
      return idx >= 0 ? idx : 0;
    }

    function filteredRows() {
      const filter = els.filter.value;
      const q = els.search.value.trim().toLowerCase();
      return rows.map((row, index) => ({row, index})).filter(({row}) => {
        const label = row.face_present || '';
        if (filter === 'unlabeled' && label) return false;
        if (['yes', 'no', 'uncertain'].includes(filter) && label !== filter) return false;
        if (q) {
          const hay = [row.sample_id, row.source_path, row.local_video_path, row.notes].join(' ').toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });
    }

    function render() {
      if (!rows.length) return;
      const row = rows[current];
      els.video.src = `/media/${encodeURIComponent(row.sample_id)}`;
      els.sampleId.textContent = `${row.sample_id} (${current + 1}/${rows.length})`;
      els.sourcePath.textContent = row.source_path || '';
      els.videoPath.textContent = row.local_video_path || '';
      els.labeler.value = row.labeler || localStorage.getItem('faceBlurLabeler') || '';
      els.notes.value = row.notes || '';
      setSelected(row.face_present || '');
      renderSummary();
      renderList();
    }

    function renderSummary() {
      const counts = {yes: 0, no: 0, uncertain: 0, unlabeled: 0};
      for (const row of rows) {
        if (row.face_present === 'yes') counts.yes += 1;
        else if (row.face_present === 'no') counts.no += 1;
        else if (row.face_present === 'uncertain') counts.uncertain += 1;
        else counts.unlabeled += 1;
      }
      els.summary.textContent = `${rows.length} total · ${counts.yes} yes · ${counts.no} no · ${counts.uncertain} uncertain · ${counts.unlabeled} unlabeled`;
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
        const label = row.face_present || 'unlabeled';
        badge.className = `badge ${row.face_present || ''}`;
        badge.textContent = label;
        btn.append(num, name, badge);
        els.list.append(btn);
      }
    }

    function setSelected(label) {
      for (const [btn, value] of [[els.yesBtn, 'yes'], [els.noBtn, 'no'], [els.uncertainBtn, 'uncertain']]) {
        btn.classList.toggle('selected', label === value);
      }
    }

    async function save(label, advance = false) {
      const row = rows[current];
      const labeler = els.labeler.value.trim();
      if (labeler) localStorage.setItem('faceBlurLabeler', labeler);
      const payload = {
        sample_id: row.sample_id,
        face_present: label,
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
      setSelected(label);
      renderSummary();
      renderList();
      if (advance) next();
    }

    function queueNotesSave() {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => save(rows[current].face_present || '', false), 450);
    }

    function next() {
      if (current < rows.length - 1) {
        current += 1;
        render();
      }
    }

    function previous() {
      if (current > 0) {
        current -= 1;
        render();
      }
    }

    document.getElementById('yesBtn').addEventListener('click', () => save('yes', true));
    document.getElementById('noBtn').addEventListener('click', () => save('no', true));
    document.getElementById('uncertainBtn').addEventListener('click', () => save('uncertain', true));
    document.getElementById('clearBtn').addEventListener('click', () => save('', false));
    document.getElementById('nextBtn').addEventListener('click', next);
    document.getElementById('prevBtn').addEventListener('click', previous);
    els.notes.addEventListener('input', queueNotesSave);
    els.labeler.addEventListener('change', () => save(rows[current].face_present || '', false));
    els.filter.addEventListener('change', renderList);
    els.search.addEventListener('input', renderList);

    document.addEventListener('keydown', event => {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (event.key === 'y' || event.key === 'Y') save('yes', true);
      else if (event.key === 'n' || event.key === 'N') save('no', true);
      else if (event.key === 'u' || event.key === 'U') save('uncertain', true);
      else if (event.key === 'ArrowRight') next();
      else if (event.key === 'ArrowLeft') previous();
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
        }

    def update_label(
        self,
        sample_id: str,
        *,
        face_present: str,
        labeler: str,
        notes: str,
    ) -> dict:
        if face_present not in ALLOWED_LABELS:
            raise ValueError(f"face_present must be one of {sorted(ALLOWED_LABELS)!r}")
        with self.lock:
            self._load_locked()
            for row in self.rows:
                if row.get("sample_id") == sample_id:
                    row["face_present"] = face_present
                    row["labeler"] = labeler
                    row["notes"] = notes
                    self._write_locked()
                    return self.snapshot()
        raise KeyError(f"sample_id not found: {sample_id}")

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
    counts = {"total": len(rows), "yes": 0, "no": 0, "uncertain": 0, "unlabeled": 0}
    for row in rows:
        label = row.get("face_present") or ""
        if label in ("yes", "no", "uncertain"):
            counts[label] += 1
        else:
            counts["unlabeled"] += 1
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
