"""QA columns of the episode index.

``qa_results`` holds one entry per check that ran on the episode: the check's
version, when it ran, whether it applied, its error, findings and skips. Finding
details beyond category, test and key are a JSON string. The other ``qa_*``
columns summarize the entries whose version matches the current check version:

- ``qa_run_at``: latest run of any check
- ``qa_error``: errors, as ``check: message``, joined by ``;``
- ``qa_video_checked``: whether any video check applied
- ``qa_checked``: checks that applied
- ``qa_findings`` and ``qa_skips``: all findings and skips
- ``qa_flagged``: whether any finding fired
"""

from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from oopsie_backend.index.store import KEY
from oopsie_backend.qa.checks.base import VIDEO, Check
from oopsie_backend.qa.engine import to_builtin

_BASE = ("category", "test", "key")
QA_FINDING = pl.Struct({"category": pl.String, "test": pl.String, "key": pl.String, "details": pl.String})
QA_SKIP = pl.Struct({"category": pl.String, "test": pl.String, "key": pl.String, "reason": pl.String})
QA_RESULT = pl.Struct(
    {
        "check": pl.String,
        "version": pl.String,
        "run_at": pl.Datetime("us", "UTC"),
        "ran": pl.Boolean,
        "error": pl.String,
        "findings": pl.List(QA_FINDING),
        "skips": pl.List(QA_SKIP),
    }
)
QA_SCHEMA = {
    KEY: pl.String,
    "qa_results": pl.List(QA_RESULT),
    "qa_run_at": pl.Datetime("us", "UTC"),
    "qa_error": pl.String,
    "qa_video_checked": pl.Boolean,
    "qa_checked": pl.List(pl.String),
    "qa_findings": pl.List(QA_FINDING),
    "qa_skips": pl.List(QA_SKIP),
    "qa_flagged": pl.Boolean,
}
QA_COLUMNS = [c for c in QA_SCHEMA if c != KEY]


def result_entry(name: str, version: str, run_at: datetime, result: dict) -> dict:
    """An index entry from an ``engine.run_episode`` result."""
    return {
        "check": name,
        "version": version,
        "run_at": run_at,
        "ran": result["ran"],
        "error": result["error"],
        "findings": [
            {
                **{k: str(f.get(k)) for k in _BASE},
                "details": json.dumps(
                    {k: v for k, v in f.items() if k not in _BASE}, default=to_builtin, sort_keys=True
                ),
            }
            for f in result["findings"]
        ],
        "skips": [{**{k: str(s.get(k)) for k in _BASE}, "reason": str(s.get("reason"))} for s in result["skips"]],
    }


def current_entries(entries: list[dict] | None, versions: dict[str, str]) -> list[dict]:
    """Entries of registered, enabled checks at their current version."""
    return [e for e in entries or [] if versions.get(e["check"]) == e["version"]]


def qa_row(key: str, entries: list[dict], versions: dict[str, str], checks: dict[str, Check]) -> dict:
    """The QA columns of one episode; ``entries`` are kept only for checks in ``versions``."""
    kept = [e for e in entries if e["check"] in versions]
    current = current_entries(kept, versions)
    errors = [f"{e['check']}: {e['error']}" for e in current if e["error"]]
    return {
        KEY: key,
        "qa_results": sorted(kept, key=lambda e: e["check"]),
        "qa_run_at": max((e["run_at"] for e in current), default=None),
        "qa_error": "; ".join(errors) or None,
        "qa_video_checked": any(e["ran"] and checks[e["check"]].kind == VIDEO for e in current),
        "qa_checked": sorted(e["check"] for e in current if e["ran"]),
        "qa_findings": [f for e in current for f in e["findings"]],
        "qa_skips": [s for e in current for s in e["skips"]],
        "qa_flagged": any(e["findings"] for e in current),
    }


def qa_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=QA_SCHEMA)


def finding_dict(finding: dict) -> dict:
    """A stored finding with its details expanded again."""
    return {**{k: finding[k] for k in _BASE}, **json.loads(finding["details"] or "{}")}
