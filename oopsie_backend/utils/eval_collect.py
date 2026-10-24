"""Collect annotations from returned eval bundles into an append-only store.

A record is one ``episode_annotations/<name>`` group of one returned file.
Records are keyed by ``(uid, bundle, name, content hash)``: re-collecting the
same return is a no-op, a changed annotation is added as a new revision, and
nothing already in the store is dropped. Annotations from different bundles
never replace each other, even when annotators typed the same name.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from oopsie_data_tools.annotation_tool.annotation_schema import read_annotation_attrs

from oopsie_backend.utils.eval_bundles import MANIFEST_NAME, sample_uid, source_path
from oopsie_backend.utils.qa_sampling import _video_files, repo_root

UNKNOWN_BUNDLE = "<unknown>"
ORIGIN_BUNDLE = "eval_bundle"
ORIGIN_NAME = "eval_name"


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _content_hash(attrs: dict) -> str:
    return hashlib.sha1(json.dumps(attrs, sort_keys=True).encode()).hexdigest()[:16]


def load_assignment(assignment_dir: Path) -> dict[str, dict]:
    """uid -> source episode, split and assigned annotators, from ``assign_eval_annotators`` output."""
    by_uid: dict[str, dict] = {}
    for path in sorted(assignment_dir.glob("*.json")):
        if path.name == "assignment_summary.json":
            continue
        lists = json.loads(path.read_text())
        for e in lists["episodes"]:
            uid = sample_uid(e["dataset_repo"], e["repo_path"])
            entry = by_uid.setdefault(
                uid,
                {
                    "dataset_repo": e["dataset_repo"],
                    "repo_path": e["repo_path"],
                    "split": e["split"],
                    "lab_id": e["lab_id"],
                    "task": e["task"],
                    "assigned": [],
                },
            )
            entry["assigned"].append(lists["annotator"])
    return by_uid


def _records_from_h5(f: h5py.File, source: str, bundle: str, assignment: dict[str, dict]) -> tuple[list[dict], str | None]:
    uid = _jsonable(f.attrs.get("episode_id"))
    if uid not in assignment:
        return [], f"{source}: episode_id {uid!r} is not in the assignment"
    ea = f.get("episode_annotations")
    if not isinstance(ea, h5py.Group):
        return [], None
    meta = assignment[uid]
    out = []
    for name, group in ea.items():
        if not isinstance(group, h5py.Group):
            continue
        attrs = {k: _jsonable(v) for k, v in group.attrs.items()}
        out.append(
            {
                "uid": uid,
                "dataset_repo": meta["dataset_repo"],
                "repo_path": meta["repo_path"],
                "split": meta["split"],
                "bundle": bundle,
                "name": str(name),
                "timestamp": attrs.get("timestamp"),
                "source": attrs.get("source"),
                "hash": _content_hash(attrs),
                "annotation": read_annotation_attrs(group.attrs),
                "attrs": attrs,
                "source_file": source,
            }
        )
    return out, None


def _bundle_name(manifest: dict | None, fallback: str | None) -> str:
    if manifest and manifest.get("annotator"):
        return manifest["annotator"]
    return fallback or UNKNOWN_BUNDLE


def read_returns(paths: list[Path], assignment: dict[str, dict], bundle: str | None = None) -> tuple[list[dict], list[str]]:
    """Records from returned zips, bundle folders or single h5 files."""
    records: list[dict] = []
    problems: list[str] = []

    def add(found: tuple[list[dict], str | None]) -> None:
        recs, problem = found
        records.extend(recs)
        if problem:
            problems.append(problem)

    for path in paths:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                manifest_name = next((n for n in names if n.endswith(MANIFEST_NAME)), None)
                manifest = json.loads(zf.read(manifest_name)) if manifest_name else None
                name = _bundle_name(manifest, bundle)
                for member in sorted(n for n in names if n.endswith(".h5")):
                    with h5py.File(BytesIO(zf.read(member)), "r") as f:
                        add(_records_from_h5(f, f"{path}:{member}", name, assignment))
            continue
        files = sorted(path.rglob("*.h5")) if path.is_dir() else [path]
        manifest_path = next(
            (p / MANIFEST_NAME for p in [path, *path.parents] if (p / MANIFEST_NAME).is_file()), None
        )
        manifest = json.loads(manifest_path.read_text()) if manifest_path else None
        name = _bundle_name(manifest, bundle)
        for file in files:
            with h5py.File(file, "r") as f:
                add(_records_from_h5(f, str(file), name, assignment))
    return records, problems


def _record_key(r: dict) -> tuple:
    return r["uid"], r["bundle"], r["name"], r["hash"]


def read_store(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def merge_into_store(path: Path, new: list[dict]) -> tuple[int, int]:
    """Append records not yet in the store; return ``(added, already_present)``."""
    known = {_record_key(r) for r in read_store(path)}
    added = []
    for r in new:
        k = _record_key(r)
        if k not in known:
            known.add(k)
            added.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in added:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(added), len(new) - len(added)


def current_annotations(store: list[dict]) -> dict[str, list[dict]]:
    """uid -> latest revision of every ``(bundle, name)`` annotation of that episode."""
    latest: dict[tuple, dict] = {}
    for r in store:
        k = (r["uid"], r["bundle"], r["name"])
        if k not in latest or (r.get("timestamp") or "") >= (latest[k].get("timestamp") or ""):
            latest[k] = r
    out: dict[str, list[dict]] = defaultdict(list)
    for (uid, _, _), r in sorted(latest.items()):
        out[uid].append(r)
    return dict(out)


def coverage_report(store: list[dict], assignment: dict[str, dict]) -> dict:
    """Returned vs assigned annotations, revisions, and outcome agreement per split."""
    current = current_annotations(store)
    revisions = Counter((r["uid"], r["bundle"], r["name"]) for r in store)
    returned_pairs = {(r["uid"], r["bundle"]) for rs in current.values() for r in rs}
    missing = Counter()
    unexpected = 0
    for uid, meta in assignment.items():
        for a in meta["assigned"]:
            if (uid, a) not in returned_pairs:
                missing[a] += 1
    for uid, bundle in returned_pairs:
        if bundle not in assignment[uid]["assigned"]:
            unexpected += 1
    agreement: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for uid, rs in current.items():
        outcomes = {r["annotation"].get("outcome") for r in rs if r["annotation"].get("outcome")}
        if sum(1 for r in rs if r["annotation"].get("outcome")) >= 2:
            agreement[assignment[uid]["split"]][1] += 1
            agreement[assignment[uid]["split"]][0] += len(outcomes) == 1
    return {
        "episodes_assigned": len(assignment),
        "annotations_assigned": sum(len(m["assigned"]) for m in assignment.values()),
        "annotations_returned": len(returned_pairs),
        "missing_by_annotator": dict(sorted(missing.items())),
        "unexpected_returns": unexpected,
        "revised_annotations": sum(1 for n in revisions.values() if n > 1),
        "outcome_agreement_by_split": {s: {"agree": a, "with_2plus": n} for s, (a, n) in sorted(agreement.items())},
    }


def _group_name(r: dict, taken: set[str], same_name_bundles: set[str]) -> str:
    """``name``, or ``name__bundle`` when several bundles used the name or it is taken."""
    name = r["name"] if len(same_name_bundles) == 1 else f"{r['name']}__{r['bundle']}"
    base, i = name, 2
    while name in taken:
        name = f"{base}__{i}"
        i += 1
    return name


def _origin(group: h5py.Group) -> tuple[str, str] | None:
    if ORIGIN_BUNDLE in group.attrs and ORIGIN_NAME in group.attrs:
        return _jsonable(group.attrs[ORIGIN_BUNDLE]), _jsonable(group.attrs[ORIGIN_NAME])
    return None


def _group_hash(group: h5py.Group) -> str:
    return _content_hash(
        {k: _jsonable(v) for k, v in group.attrs.items() if k not in (ORIGIN_BUNDLE, ORIGIN_NAME)}
    )


def apply_to_copies(
    store: list[dict], assignment: dict[str, dict], data_dir: Path, out_dir: Path
) -> tuple[int, int, int]:
    """Write source episodes with all current eval annotations; return ``(episodes, added, revised)``.

    Copies keep the repo layout under ``out_dir/<repo>/`` with their videos.
    Groups written here carry ``eval_bundle`` and ``eval_name`` attributes; a
    newer revision of the same bundle and name rewrites only that group. Other
    groups, including the lab's own annotations, are never modified.
    """
    added = revised = 0
    current = current_annotations(store)
    for uid, rs in current.items():
        meta = assignment[uid]
        src = source_path(data_dir, meta["dataset_repo"], meta["repo_path"])
        repo = meta["dataset_repo"].split("/", 1)[1]
        dst = out_dir / repo / meta["repo_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            shutil.copyfile(src, dst)
            for _, (video, rel) in _video_files(src, repo_root(src, meta["repo_path"])).items():
                target = out_dir / repo / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file():
                    shutil.copyfile(video, target)
        bundles_per_name: dict[str, set[str]] = defaultdict(set)
        for r in rs:
            bundles_per_name[r["name"]].add(r["bundle"])
        with h5py.File(dst, "r+") as f:
            ea = f.require_group("episode_annotations")
            groups = [g for g in ea.values() if isinstance(g, h5py.Group)]
            by_origin = {_origin(g): g for g in groups if _origin(g)}
            hashes = {_group_hash(g) for g in groups}
            for r in rs:
                group = by_origin.get((r["bundle"], r["name"]))
                if group is not None:
                    if _group_hash(group) == r["hash"]:
                        continue
                    for k in list(group.attrs):
                        del group.attrs[k]
                    revised += 1
                elif r["hash"] in hashes:
                    continue
                else:
                    group = ea.create_group(_group_name(r, set(ea.keys()), bundles_per_name[r["name"]]))
                    added += 1
                for k, v in r["attrs"].items():
                    group.attrs[k] = v
                group.attrs[ORIGIN_BUNDLE] = r["bundle"]
                group.attrs[ORIGIN_NAME] = r["name"]
                hashes.add(r["hash"])
    print(
        f"wrote {len(current)} episode(s) to {out_dir}: {added} annotation group(s) added, {revised} revised",
        file=sys.stderr,
    )
    return len(current), added, revised
