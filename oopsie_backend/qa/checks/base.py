"""The ``@check`` decorator and the registry it fills.

Every QA check is a function registered with ``@check(category, test, kind)`` in
its own file, ``qa/checks/<category>/<test>.py``; ``qa/checks/registry.py``
imports each of those files by name. The function gets the episode
(``qa.engine.EpisodeContext``), the full QA config and its ``Check``, and returns
``{"findings", "skips", "checked"}``. ``category.run_category`` runs the shared
category logic (``qa/action_qa.py`` or ``qa/video_checks.py``) for one check.

``kind`` names the input a check needs: ``action`` (actions, robot states and
the robot profile), ``metadata`` (root attributes and the profile) or ``video``
(decoded camera videos plus actions and states).

A check's version hashes its file, the logic module of its kind, its config
section (``configs/action_qa_config.yaml``, ``<category>.<test>``), the sections
in ``extra_sections`` and, for video checks, ``video_settings``. Results are
recomputed only when that version changes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ACTION = "action"
METADATA = "metadata"
VIDEO = "video"

QA_DIR = Path(__file__).resolve().parent.parent
LOGIC_MODULES = {
    ACTION: (QA_DIR / "action_qa.py",),
    METADATA: (QA_DIR / "action_qa.py",),
    VIDEO: (QA_DIR / "video_checks.py", QA_DIR / "action_qa.py"),
}


@dataclass(frozen=True)
class Check:
    category: str
    test: str
    kind: str
    fn: Callable[..., dict]
    extra_sections: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.category}/{self.test}"

    def enabled(self, config: dict) -> bool:
        return bool(config[self.category][self.test]["enabled"])

    def version(self, config: dict) -> str:
        h = hashlib.sha256()
        for path in (Path(inspect.getsourcefile(self.fn)), *LOGIC_MODULES[self.kind]):
            h.update(path.read_bytes())
        sections = {s: config[self.category][s] for s in (self.test, *self.extra_sections)}
        params = {s: {k: v for k, v in c.items() if k != "enabled"} for s, c in sections.items()}
        if self.kind == VIDEO:
            params["video_settings"] = config["video_settings"]
        h.update(json.dumps(params, sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]


CHECKS: dict[str, Check] = {}


def check(category: str, test: str, kind: str, extra_sections: tuple[str, ...] = ()):
    """Register the decorated function as the QA check ``category/test``."""

    def register(fn: Callable[..., dict]) -> Callable[..., dict]:
        c = Check(category, test, kind, fn, extra_sections)
        if c.name in CHECKS:
            raise ValueError(f"check {c.name} registered twice")
        CHECKS[c.name] = c
        return fn

    return register
