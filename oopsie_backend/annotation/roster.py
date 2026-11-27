"""Annotation bundles and the roster of who holds them.

A bundle is a fixed, ordered list of episodes of one kind, ``eval`` or ``train``, written once to
``bundles/<id>.json`` and never changed. ``bundles/roster.jsonl`` is an append-only log of holdings:
each line gives a bundle to an annotator under a slot, the ``batch_<n>`` folder the bundle is
published in inside the annotator's folder. A bundle's last line names its current holder. Earlier
lines stay valid, so returns from a former holder are still accepted. Slots are never reused within
a folder, and an annotator never holds two bundles that share an episode.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from oopsie_backend.annotation.bundles import folder_name, sample_uid
from oopsie_backend.common.io import now_timestamps
from oopsie_backend.index.store import locked

EVAL = "eval"
TRAIN = "train"
KINDS = (EVAL, TRAIN)
ROSTER_NAME = "roster.jsonl"
SLOT_RE = re.compile(r"batch_(\d+)")
BUNDLE_RE = re.compile(r"b\d+")
ENTRY_FIELDS = ("dataset_repo", "repo_path", "lab_id", "robot_name", "robot", "task", "split")


@dataclass(frozen=True)
class Holding:
    bundle: str
    annotator: str
    slot: str
    time: str
    note: str = ""

    @property
    def folder(self) -> str:
        return folder_name(self.annotator)


def _key(e: dict) -> tuple[str, str]:
    return e["dataset_repo"], e["repo_path"]


def _entry(e: dict, position: int) -> dict:
    return {"position": position, **{k: e.get(k) for k in ENTRY_FIELDS}} | (
        {"calibration": True} if e.get("calibration") else {}
    )


class Registry:
    """The bundles and roster in ``root`` (``<data_dir>/annotation/bundles``).

    Changes happen inside ``writing``, which locks the roster and rereads it. A bundle file the
    roster does not name, left by a write that stopped before its roster line, is ignored.
    """

    def __init__(self, root: Path):
        self.root = root
        self._locked = False
        self._orphans: set[str] = set()
        self._load()

    def _load(self) -> None:
        files = {p.stem: p for p in sorted(self.root.glob("b*.json")) if BUNDLE_RE.fullmatch(p.stem)}
        roster = self.root / ROSTER_NAME
        lines = roster.read_text().splitlines() if roster.is_file() else []
        self.holdings = [Holding(**json.loads(line)) for line in lines if line.strip()]
        for h in self.holdings:
            if h.bundle not in files:
                raise ValueError(f"{roster} names unknown bundle {h.bundle}")
        held = {h.bundle for h in self.holdings}
        orphans = set(files) - held
        if orphans - self._orphans:
            print(f"{self.root}: ignoring bundle file(s) the roster does not name: {', '.join(sorted(orphans))}", file=sys.stderr)
        self._orphans = orphans
        self.bundles: dict[str, dict] = {b: json.loads(files[b].read_text()) for b in files if b in held}
        self._last_id = max((int(b[1:]) for b in files), default=0)

    @contextmanager
    def writing(self) -> Iterator[None]:
        """Hold the roster lock, with the registry reread from disk on entry; nested calls reuse it."""
        if self._locked:
            yield
            return
        with locked(self.root / ROSTER_NAME):
            self._locked = True
            try:
                self._load()
                yield
            finally:
                self._locked = False

    @property
    def exists(self) -> bool:
        return (self.root / ROSTER_NAME).is_file()

    def files(self) -> list[Path]:
        return [self.root / ROSTER_NAME, *(self.root / f"{b}.json" for b in sorted(self.bundles))]

    def current(self) -> list[Holding]:
        """The current holding of every bundle, ordered by bundle id."""
        last = {h.bundle: h for h in self.holdings}
        return [last[b] for b in sorted(last)]

    def holder(self, bundle: str) -> Holding:
        return next(h for h in reversed(self.holdings) if h.bundle == bundle)

    def by_slot(self) -> dict[tuple[str, str], Holding]:
        """``(folder, slot)`` -> holding, over the whole roster."""
        return {(h.folder, h.slot): h for h in self.holdings}

    def annotators(self) -> set[str]:
        return {h.annotator for h in self.holdings}

    def seen(self, annotator: str) -> set[tuple[str, str]]:
        """Episodes of every bundle the annotator holds or held."""
        held = {h.bundle for h in self.holdings if h.annotator == annotator}
        return {_key(e) for b in held for e in self.bundles[b]["episodes"]}

    def episodes(self, kind: str | None = None) -> dict[tuple[str, str], list[str]]:
        """Episode -> ids of the bundles of ``kind`` (any kind if None) that contain it."""
        out: dict[tuple[str, str], list[str]] = {}
        for bid, b in sorted(self.bundles.items()):
            if kind is None or b["kind"] == kind:
                for e in b["episodes"]:
                    out.setdefault(_key(e), []).append(bid)
        return out

    def _check_annotator(self, annotator: str, episodes: list[dict]) -> None:
        folder = folder_name(annotator)
        if not folder:
            raise ValueError(f"annotator name {annotator!r} has no usable folder name")
        clash = sorted({h.annotator for h in self.holdings if h.folder == folder and h.annotator != annotator})
        if clash:
            raise ValueError(f"annotator {annotator!r} and {clash[0]!r} map to folder {folder!r}")
        overlap = self.seen(annotator) & {_key(e) for e in episodes}
        if overlap:
            raise ValueError(f"{annotator!r} already holds or held {len(overlap)} of these episodes")

    def _next_slot(self, annotator: str) -> str:
        folder = folder_name(annotator)
        used = [int(m.group(1)) for h in self.holdings if h.folder == folder and (m := SLOT_RE.fullmatch(h.slot))]
        return f"batch_{max(used, default=0) + 1}"

    def _append(self, holding: Holding) -> None:
        with (self.root / ROSTER_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(holding), ensure_ascii=False) + "\n")
        self.holdings.append(holding)

    def add(
        self, kind: str, episodes: list[dict], annotator: str, note: str, slot: str | None = None, **meta
    ) -> Holding:
        """Write a new bundle of ``episodes``, at positions 1, 2, ... in their order, and give it to ``annotator``.

        The bundle goes to ``slot``, which must be unused in the annotator's folder, or else to the next slot.
        """
        with self.writing():
            if kind not in KINDS:
                raise ValueError(f"unknown bundle kind {kind!r}")
            keys = [_key(e) for e in episodes]
            if not episodes or len(set(keys)) != len(keys):
                raise ValueError("a bundle needs at least one episode and no repeated episodes")
            self._check_annotator(annotator, episodes)
            if slot is not None:
                if not SLOT_RE.fullmatch(slot):
                    raise ValueError(f"slot {slot!r} is not batch_<n>")
                if (folder_name(annotator), slot) in self.by_slot():
                    raise ValueError(f"{annotator!r} already has {slot}")
            bid = f"b{self._last_id + 1:04d}"
            bundle = {
                "bundle": bid,
                "kind": kind,
                "created_at": now_timestamps()["utc"],
                **meta,
                "episodes": [_entry(e, i) for i, e in enumerate(episodes, 1)],
            }
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / f"{bid}.json").open("x") as fh:
                fh.write(json.dumps(bundle, indent=2, ensure_ascii=False))
            self.bundles[bid] = bundle
            self._last_id += 1
            holding = Holding(bid, annotator, slot or self._next_slot(annotator), bundle["created_at"], note)
            self._append(holding)
            return holding

    def reassign(self, bundle: str, annotator: str, note: str = "") -> Holding:
        """Give ``bundle`` to ``annotator`` under a new slot of their folder."""
        with self.writing():
            if bundle not in self.bundles:
                raise ValueError(f"unknown bundle {bundle!r}")
            if self.holder(bundle).annotator == annotator:
                raise ValueError(f"{annotator!r} already holds {bundle}")
            self._check_annotator(annotator, self.bundles[bundle]["episodes"])
            holding = Holding(bundle, annotator, self._next_slot(annotator), now_timestamps()["utc"], note)
            self._append(holding)
            return holding

    def assignment(self, kind: str) -> dict[str, dict]:
        """uid -> source episode, split, current holders (``assigned``) and every holder (``held``) over bundles of ``kind``."""
        holders: dict[str, list[str]] = {}
        for h in self.holdings:
            holders.setdefault(h.bundle, []).append(h.annotator)
        out: dict[str, dict] = {}
        for bid, b in sorted(self.bundles.items()):
            if b["kind"] != kind or bid not in holders:
                continue
            for e in b["episodes"]:
                uid = sample_uid(e["dataset_repo"], e["repo_path"])
                meta = out.setdefault(
                    uid,
                    {k: e[k] for k in ("dataset_repo", "repo_path", "split", "lab_id", "task")}
                    | {"assigned": [], "held": []},
                )
                meta["assigned"].append(holders[bid][-1])
                meta["held"].extend(a for a in holders[bid] if a not in meta["held"])
        return out

    def positions(self, holding: Holding) -> dict[int, str]:
        """Position -> uid of the holding's bundle."""
        return {
            e["position"]: sample_uid(e["dataset_repo"], e["repo_path"])
            for e in self.bundles[holding.bundle]["episodes"]
        }


def adopt(root: Path, sources: list[tuple[str, str, Path]]) -> Registry:
    """Create ``root`` from per-annotator assignment folders, ``(kind, slot, assignment_dir)`` each.

    Every non-empty annotator list becomes one bundle with its positions kept, held by its annotator under ``slot``.
    """
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    registry = Registry(tmp)
    for kind, slot, assignment_dir in sources:
        for path in sorted(assignment_dir.glob("*.json")):
            if path.name == "assignment_summary.json":
                continue
            lists = json.loads(path.read_text())
            if not lists["episodes"]:
                print(f"adopt: {path} lists no episodes; no bundle", file=sys.stderr)
                continue
            if [e["position"] for e in lists["episodes"]] != list(range(1, len(lists["episodes"]) + 1)):
                raise ValueError(f"{path}: positions are not 1..{len(lists['episodes'])} in list order")
            registry.add(
                kind, lists["episodes"], lists["annotator"], f"adopted from {assignment_dir.name}",
                slot=slot, proposal=lists.get("proposal"),
            )
    os.replace(tmp, root)
    return Registry(root)
