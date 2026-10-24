"""Timestamps and JSON output shared by all stages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

CENTRAL_TZ = ZoneInfo("America/Chicago")


def now_timestamps() -> dict[str, str]:
    """Return current time as UTC and US Central (CDT/CST) ISO strings."""
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    now_central = now_utc.astimezone(CENTRAL_TZ)
    return {
        "utc": now_utc.isoformat().replace("+00:00", "Z"),
        "cdt": now_central.isoformat(),
    }


def write_stats_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
