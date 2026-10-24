"""Render a self-contained public stats HTML report using the docs dashboard."""

from __future__ import annotations

import json
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
DASHBOARD_INCLUDE = _REPO_ROOT / "oopsie-docs/docs/_includes/stats-dashboard.html"


def _dashboard_fragment() -> str:
    if not DASHBOARD_INCLUDE.is_file():
        raise FileNotFoundError(
            f"Shared stats dashboard not found: {DASHBOARD_INCLUDE}"
        )
    return DASHBOARD_INCLUDE.read_text(encoding="utf-8")


def render_public_report_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    fragment = _dashboard_fragment()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OopsieData — public stats</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    }}
    #stats-root {{
      max-width: 70rem;
      margin: 0 auto;
      padding: 1.25rem 1rem 3rem;
    }}
  </style>
</head>
<body>
  <script>window.__OOPSIE_STATS__ = {payload};</script>
  {fragment}
</body>
</html>
"""


def render_public_report(data: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_public_report_html(data), encoding="utf-8")


def load_public_stats(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
