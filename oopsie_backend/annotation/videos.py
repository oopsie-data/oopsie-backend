"""The videos an annotator watched, which decide whether an annotation matches an index episode.

A bundle slot ``<folder>/<slot>/`` in the batch repo holds ``episodes/NNNN_<cam>.mp4`` for every
position. ``bundle_videos`` reads the sha256 of these files from the Hub listing of the batch repo,
at its main branch or, for slots removed since, at the newest commit that still has them. Slots are
never reused within a folder, so a slot's hashes never change; they are kept in
``<returns_dir>/bundle_videos.json`` and read from the Hub only once.

An annotation matches an episode while every video the annotator watched has the same sha256 as
the episode's video of that camera in the index (``videos_match``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi

from oopsie_backend.common.hf import hub_files

CACHE_NAME = "bundle_videos.json"
VIDEO_RE = re.compile(r"^([^/]+)/([^/]+)/episodes/(\d+)_(.+)\.mp4$")

Slot = tuple[str, str]
# position -> camera -> sha256
SlotVideos = dict[int, dict[str, str]]


def returned_position(path: str | Path) -> tuple[Slot, int] | None:
    """``((folder, slot), position)`` of a returned ``.../<folder>/<slot>/episodes/NNNN.h5``."""
    parts = PurePosixPath(Path(path).as_posix()).parts
    if len(parts) < 4 or parts[-2] != "episodes" or not parts[-1].removesuffix(".h5").isdigit():
        return None
    return (parts[-4], parts[-3]), int(parts[-1].removesuffix(".h5"))


def _slot_videos(listing: dict[str, dict]) -> dict[Slot, SlotVideos]:
    out: dict[Slot, SlotVideos] = {}
    for path, entry in listing.items():
        m = VIDEO_RE.match(path)
        if m and entry.get("sha256"):
            folder, slot, position, cam = m.groups()
            out.setdefault((folder, slot), {}).setdefault(int(position), {})[cam] = entry["sha256"]
    return out


def _read_cache(path: Path) -> dict[Slot, SlotVideos]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {
        tuple(key.split("/", 1)): {int(p): cams for p, cams in positions.items()}
        for key, positions in raw.items()
    }


def _write_cache(path: Path, videos: dict[Slot, SlotVideos]) -> None:
    raw = {f"{f}/{s}": {str(p): cams for p, cams in sorted(v.items())} for (f, s), v in sorted(videos.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n")


def bundle_videos(repo_id: str, slots: set[Slot], cache: Path, api: HfApi | None = None) -> dict[Slot, SlotVideos]:
    """The video hashes of ``slots``; a slot the batch repo never had, or that cannot be read, is left out."""
    known = _read_cache(cache)
    missing = slots - known.keys()
    if missing:
        api = api or HfApi()
        try:
            found = _slot_videos(hub_files(api, repo_id))
            known |= {s: found[s] for s in missing & found.keys()}
            missing -= found.keys()
            if missing:
                for commit in api.list_repo_commits(repo_id, repo_type="dataset"):
                    found = _slot_videos(hub_files(api, repo_id, revision=commit.commit_id))
                    known |= {s: found[s] for s in missing & found.keys()}
                    missing -= found.keys()
                    if not missing:
                        break
        except Exception as exc:
            print(f"bundle videos: cannot read {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        _write_cache(cache, known)
    if missing:
        names = ", ".join(f"{f}/{s}" for f, s in sorted(missing))
        print(f"bundle videos: no videos found for {len(missing)} slot(s): {names}", file=sys.stderr)
    return {s: known[s] for s in slots if s in known}


def watched(videos: dict[Slot, SlotVideos], path: str | Path) -> dict[str, str] | None:
    """camera -> sha256 of the videos shown with the returned file at ``path``."""
    found = returned_position(path)
    if found is None:
        return None
    slot, position = found
    return videos.get(slot, {}).get(position)


def videos_match(seen: dict[str, str] | None, videos: list[dict] | None) -> bool:
    """Whether every video in ``seen`` (camera -> sha256) is unchanged in the index ``videos``."""
    if not seen:
        return False
    current = {v["camera"]: v["sha256"] for v in videos or []}
    return all(current.get(cam) == sha for cam, sha in seen.items())
