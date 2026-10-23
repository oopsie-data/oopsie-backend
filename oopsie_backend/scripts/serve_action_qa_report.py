"""Serve the action QA report with its evidence panel: the report page, the episodes'
camera videos (with seeking), and per-episode evidence read on demand.

Every request needs the password (HTTP basic auth, any username), so the page can be
shared, e.g. through ngrok; it serves lab videos, so don't run it without one.

Usage:
    OOPSIE_QA_PASSWORD=... uv run python -m oopsie_backend.scripts.serve_action_qa_report
    # options: --report reports/action_qa.json --data-dir <episodes dir> --port 8010
    # then open http://localhost:8010/  (or: ngrok http 8010)
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import mimetypes
import os
import re
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from oopsie_backend.utils.action_qa_config import CONFIG as ACTION_QA_CONFIG
from oopsie_backend.utils.qa_evidence import episode_evidence

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov"}  # the only files served from the data dir
CHUNK = 1 << 16


def make_handler(report_path: Path, data_dir: Path, password: str):
    html_path = report_path.with_suffix(".html")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings_by_episode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ep in report.get("flagged_episodes", []):
        findings_by_episode[(ep["dataset_repo"], ep["repo_path"])].extend(ep["findings"])
    ignore = ACTION_QA_CONFIG["video_settings"]["ignore_cameras"]
    root = data_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet terminal
            pass

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            if not header.startswith("Basic "):
                return False
            try:
                _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            except Exception:
                return False
            return hmac.compare_digest(given.encode(), password.encode())

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorized():
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="OopsieData QA report"')
                self.end_headers()
                return
            url = urlparse(self.path)
            if url.path in ("/", "/index.html", "/" + html_path.name):
                return self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/evidence":
                return self._evidence(parse_qs(url.query))
            if url.path.startswith("/data/"):
                return self._video(unquote(url.path[len("/data/"):]))
            self._send(404, b"not found", "text/plain")

        def _evidence(self, q: dict) -> None:
            get = lambda k: (q.get(k) or [""])[0]
            dataset, repo_path, category, test = get("dataset"), get("repo_path"), get("category"), get("test")
            findings = [f for f in findings_by_episode.get((dataset, repo_path), [])
                        if f.get("category") == category and f.get("test") == test]
            try:
                body = episode_evidence(root, dataset, repo_path, category, test, findings, ignore, "data/")
                body["findings"] = findings
                code = 200
            except Exception as exc:
                body, code = {"error": f"{type(exc).__name__}: {exc}"}, 400
            self._send(code, json.dumps(body).encode(), "application/json")

        def _video(self, rel: str) -> None:
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or path.suffix.lower() not in VIDEO_SUFFIXES or not path.is_file():
                return self._send(404, b"not found", "text/plain")
            size = path.stat().st_size
            start, end, code = 0, size - 1, 200
            if m := RANGE.match(self.headers.get("Range", "")):
                a, b = m.groups()
                if a:
                    start, end = int(a), min(int(b) if b else size - 1, size - 1)
                else:  # "bytes=-N": the last N bytes
                    start = max(size - int(b or 0), 0)
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):  # browser cancelled
                        break
                    remaining -= len(chunk)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=ACTION_QA_CONFIG["run"]["output"],
                        help="action_qa.json (its .html must sit next to it). Default: run.output from the config.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Where the episodes live. Default: the data dir recorded in the report.")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--password", default=None, help="Default: the OOPSIE_QA_PASSWORD environment variable.")
    args = parser.parse_args(argv)

    password = args.password or os.environ.get("OOPSIE_QA_PASSWORD")
    if not password:
        print("error: set a password with --password or OOPSIE_QA_PASSWORD (the page serves lab videos)", file=sys.stderr)
        return 1
    report_path = args.report.resolve()
    if not report_path.is_file() or not report_path.with_suffix(".html").is_file():
        print(f"error: need {report_path} and {report_path.with_suffix('.html')}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data_dir = args.data_dir or Path(report["config_snapshot"]["run"]["data_dir"])
    if not data_dir.is_dir():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    # Localhost only: share through a tunnel (ngrok http <port>), which forwards to it.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(report_path, data_dir, password))
    print(f"Serving {report_path.with_suffix('.html').name} at http://localhost:{args.port}/ (password required; Ctrl+C to stop)")
    print(f"Videos and evidence from {data_dir}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
