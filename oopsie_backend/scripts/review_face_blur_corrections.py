"""Serve a side-by-side manual review UI for corrected face blurs."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DECISIONS = {"approved", "needs_more_blur", "unsure", ""}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face-blur correction review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#101318;color:#edf1f7}header{padding:12px 18px;background:#1a2029;position:sticky;top:0;z-index:3}.top{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.progress{font-weight:700}select,button,input,textarea{font:inherit}button{padding:8px 12px;border:0;border-radius:6px;cursor:pointer}.nav{background:#3a4658;color:white}.approve{background:#218c52;color:white}.reject{background:#bd3b38;color:white}.unsure{background:#c38919;color:#111}.selected{outline:3px solid white}.meta{padding:10px 18px;color:#cbd5e1;line-height:1.5}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:0 10px}.panel{background:#080a0d;padding:8px}.panel h3{margin:0 0 7px;text-align:center}.panel video{width:100%;max-height:62vh;background:black}.controls,.decisions{display:flex;gap:8px;justify-content:center;align-items:center;flex-wrap:wrap;padding:10px}.notes{display:block;width:min(900px,90%);height:55px;margin:0 auto 12px;background:#1a2029;color:white;border:1px solid #465468;border-radius:6px;padding:8px}@media(max-width:1000px){.videos{grid-template-columns:1fr}.panel video{max-height:45vh}}
</style></head><body>
<header><div class="top"><span class="progress" id="progress"></span><button class="nav" id="prev">← Previous</button><button class="nav" id="next">Next →</button><label>Filter <select id="filter"><option value="all">All</option><option value="unreviewed" selected>Unreviewed</option><option value="needs_more_blur">Needs more blur</option><option value="approved">Approved</option><option value="unsure">Unsure</option></select></label><label><input type="checkbox" id="sync" checked> synchronize players</label></div></header>
<div class="meta" id="meta"></div>
<div class="videos"><div class="panel"><h3>Raw original</h3><video id="raw" controls preload="metadata"></video></div><div class="panel"><h3>Previous branch output</h3><video id="previous" controls preload="metadata"></video></div><div class="panel"><h3>Corrected output</h3><video id="corrected" controls preload="metadata"></video></div></div>
<div class="controls"><button class="nav" id="play">Play all</button><button class="nav" id="pause">Pause all</button><button class="nav" id="back">−5 seconds</button><button class="nav" id="forward">+5 seconds</button></div>
<textarea class="notes" id="notes" placeholder="Optional review notes"></textarea>
<div class="decisions"><button class="approve" id="approve">Y — Approve: every face is obscured</button><button class="reject" id="reject">N — Needs more blur</button><button class="unsure" id="unclear">U — Unsure</button></div>
<script>
let all=[],shown=[],index=0;const players=['raw','previous','corrected'].map(x=>document.getElementById(x));
function current(){return shown[index]}
function filterRows(){const f=document.getElementById('filter').value;shown=all.filter(r=>f==='all'||(f==='unreviewed'?!r.review_decision:r.review_decision===f));if(index>=shown.length)index=Math.max(0,shown.length-1);render()}
function render(){const r=current();document.getElementById('progress').textContent=shown.length?`${index+1}/${shown.length} shown · ${all.filter(x=>x.review_decision).length}/${all.length} reviewed`:`0 shown · ${all.filter(x=>x.review_decision).length}/${all.length} reviewed`;if(!r){document.getElementById('meta').textContent='No videos match this filter.';players.forEach(v=>v.removeAttribute('src'));return}document.getElementById('meta').innerHTML=`<b>${r.source_repo}</b> · ${r.sample_id}<br>${r.source_path}<br>Reason: ${r.suspected_reason}<br>Masked frames: ${r.old_masked_frame_count} previous → ${r.corrected_masked_frame_count} corrected; fallback temporal regions: ${r.fallback_temporal_region_count}`;players.forEach((v,i)=>{v.src=`/video?sample_id=${encodeURIComponent(r.sample_id)}&kind=${['raw','previous','corrected'][i]}`;v.load()});document.getElementById('notes').value=r.review_notes||'';for(const [id,d] of [['approve','approved'],['reject','needs_more_blur'],['unclear','unsure']])document.getElementById(id).classList.toggle('selected',r.review_decision===d)}
async function load(){all=await (await fetch('/api/rows')).json();all.sort((a,b)=>(b.suspected_reason.includes('visually_confirmed')-a.suspected_reason.includes('visually_confirmed'))||(Number(a.old_masked_frame_count)-Number(b.old_masked_frame_count))||a.source_repo.localeCompare(b.source_repo));filterRows()}
async function save(decision){const r=current();if(!r)return;const body={sample_id:r.sample_id,review_decision:decision,review_notes:document.getElementById('notes').value};const updated=await (await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();Object.assign(all.find(x=>x.sample_id===updated.sample_id),updated);const oldId=r.sample_id;filterRows();if(current()&&current().sample_id===oldId&&index<shown.length-1)index++;render()}
function move(n){if(!shown.length)return;index=(index+n+shown.length)%shown.length;render()}
function syncFrom(source,event){if(!document.getElementById('sync').checked)return;for(const v of players)if(v!==source){if(event==='play')v.play().catch(()=>{});else if(event==='pause')v.pause();else if(Math.abs(v.currentTime-source.currentTime)>.15)v.currentTime=source.currentTime}}
players.forEach(v=>{v.addEventListener('play',()=>syncFrom(v,'play'));v.addEventListener('pause',()=>syncFrom(v,'pause'));v.addEventListener('seeked',()=>syncFrom(v,'seek'))});
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('filter').onchange=()=>{index=0;filterRows()};document.getElementById('approve').onclick=()=>save('approved');document.getElementById('reject').onclick=()=>save('needs_more_blur');document.getElementById('unclear').onclick=()=>save('unsure');document.getElementById('play').onclick=()=>players.forEach(v=>v.play().catch(()=>{}));document.getElementById('pause').onclick=()=>players.forEach(v=>v.pause());document.getElementById('back').onclick=()=>players.forEach(v=>v.currentTime=Math.max(0,v.currentTime-5));document.getElementById('forward').onclick=()=>players.forEach(v=>v.currentTime=Math.min(v.duration||Infinity,v.currentTime+5));
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key.toLowerCase()==='y')save('approved');else if(e.key.toLowerCase()==='n')save('needs_more_blur');else if(e.key.toLowerCase()==='u')save('unsure')});load();
</script></body></html>"""


class ReviewState:
    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest.resolve()
        self.history = self.manifest.with_name("review.history.jsonl")
        self.lock = threading.Lock()
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.rows = list(reader)
            self.fields = tuple(reader.fieldnames or ())
        self.by_id = {row["sample_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("review manifest contains duplicate sample IDs")
        for row in self.rows:
            for key in ("raw_video_path", "previous_video_path", "corrected_video_path"):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(f"missing {key}: {row[key]}")

    def public_rows(self) -> list[dict[str, str]]:
        with self.lock:
            return [dict(row) for row in self.rows]

    def video(self, sample_id: str, kind: str) -> Path:
        key = {
            "raw": "raw_video_path",
            "previous": "previous_video_path",
            "corrected": "corrected_video_path",
        }.get(kind)
        if key is None or sample_id not in self.by_id:
            raise KeyError((sample_id, kind))
        return Path(self.by_id[sample_id][key])

    def save(self, payload: dict) -> dict[str, str]:
        sample_id = str(payload.get("sample_id") or "")
        decision = str(payload.get("review_decision") or "")
        notes = str(payload.get("review_notes") or "")
        reviewer = str(payload.get("reviewer") or "")
        if sample_id not in self.by_id:
            raise ValueError("unknown sample_id")
        if decision not in DECISIONS:
            raise ValueError(f"invalid review_decision: {decision!r}")
        with self.lock:
            row = self.by_id[sample_id]
            before = dict(row)
            row["review_decision"] = decision
            row["review_notes"] = notes
            row["reviewed_at"] = datetime.now(timezone.utc).isoformat() if decision else ""
            row["reviewer"] = reviewer
            self._write_manifest()
            with self.history.open("a", encoding="utf-8") as handle:
                json.dump(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "sample_id": sample_id,
                        "before": before,
                        "after": row,
                    },
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
            return dict(row)

    def _write_manifest(self) -> None:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.manifest.name}.", dir=self.manifest.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(temporary, self.manifest)
        finally:
            temporary.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    state: ReviewState

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/rows":
            self._json(self.state.public_rows())
            return
        if parsed.path == "/video":
            query = parse_qs(parsed.query)
            try:
                path = self.state.video(query["sample_id"][0], query["kind"][0])
            except (KeyError, IndexError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._video(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected JSON object")
            self._json(self.state.save(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._bytes(
            json.dumps(value, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def _bytes(
        self,
        value: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            try:
                first, last = range_header[6:].split("-", 1)
                start = int(first) if first else 0
                end = min(int(last), size - 1) if last else size - 1
                if start < 0 or end < start or start >= size:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9878)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = ReviewState(args.manifest)
    handler = type("CorrectionReviewHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Reviewing {len(state.rows)} corrected videos at "
        f"http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
