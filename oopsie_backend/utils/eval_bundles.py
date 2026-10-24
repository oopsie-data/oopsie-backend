"""Package per-annotator eval assignments into zips for ``oopsie-data annotate``.

Episodes are renamed to their position (``0001.h5``, ``0001_<cam>.mp4``) so the
annotation UI, which sorts by path, shows them in assignment order. Copies are
made blind by default: existing annotations, ``source_data`` and ``source_*``
attributes are removed, the policy name in ``robot_profile`` is redacted, and
``episode_id`` is replaced by a stable hash of the source episode, shared
across bundles.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py

from oopsie_backend.utils.episode_stats import HF_ORG, _profile, _s, now_timestamps
from oopsie_backend.utils.qa_sampling import episode_videos

MANIFEST_NAME = "eval_manifest.json"
BLIND_GROUPS = ("episode_annotations", "source_data")
BLIND_ATTR_PREFIXES = ("source_",)
BLIND_ATTRS = ("previous_episode_id",)
REDACTED_POLICY = "redacted"


def sample_uid(dataset_repo: str, repo_path: str) -> str:
    return "eval-" + hashlib.sha1(f"{dataset_repo}/{repo_path}".encode()).hexdigest()[:12]


def source_path(data_dir: Path, dataset_repo: str, repo_path: str) -> Path:
    """Local h5 of an episode; ``data_dir`` may be the mirror root or its ``HF_ORG`` folder."""
    repo = dataset_repo.split("/", 1)[1]
    base = data_dir if data_dir.name == HF_ORG else data_dir / HF_ORG
    return base / repo / repo_path


def _blind_copy(src: Path, dst: Path, uid: str, video_names: dict[str, str], blind: bool) -> None:
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        group = f.require_group("observations/video_paths")
        for cam in list(group.keys()):
            del group[cam]
        for cam, name in video_names.items():
            group.create_dataset(cam, data=name, dtype=h5py.string_dtype())
        if not blind:
            return
        for name in BLIND_GROUPS:
            if name in f:
                del f[name]
        for attr in list(f.attrs):
            if attr in BLIND_ATTRS or attr.startswith(BLIND_ATTR_PREFIXES):
                del f.attrs[attr]
        f.attrs["episode_id"] = uid
        profile = _profile(_s(f.attrs.get("robot_profile")))
        if "policy_name" in profile:
            profile["policy_name"] = REDACTED_POLICY
            f.attrs["robot_profile"] = json.dumps(profile)


def write_bundle(
    annotator: str,
    entries: list[dict],
    data_dir: Path,
    out_dir: Path,
    proposal_id: str | None,
    blind: bool = True,
) -> tuple[Path, list[str]]:
    """Write ``<annotator>.zip``; return its path and the episodes that could not be packaged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{annotator}.zip"
    manifest_entries: list[dict] = []
    missing: list[str] = []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp, zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for e in entries:
            src = source_path(data_dir, e["dataset_repo"], e["repo_path"])
            if not src.is_file():
                missing.append(f"{e['dataset_repo']} :: {e['repo_path']}: h5 not found")
                continue
            videos = episode_videos(src, e["repo_path"])
            if not videos:
                missing.append(f"{e['dataset_repo']} :: {e['repo_path']}: no videos")
                continue
            stem = f"{e['position']:04d}"
            video_names = {cam: f"{stem}_{cam}.mp4" for cam in videos}
            uid = sample_uid(e["dataset_repo"], e["repo_path"])
            h5_tmp = Path(tmp) / f"{stem}.h5"
            _blind_copy(src, h5_tmp, uid, video_names, blind)
            zf.write(h5_tmp, f"{annotator}/episodes/{stem}.h5")
            h5_tmp.unlink()
            for cam, mp4 in videos.items():
                zf.write(mp4, f"{annotator}/episodes/{video_names[cam]}")
            manifest_entries.append(
                {
                    "position": e["position"],
                    "uid": uid,
                    "bundle_path": f"episodes/{stem}.h5",
                    "video_files": {cam: f"episodes/{n}" for cam, n in video_names.items()},
                    "dataset_repo": e["dataset_repo"],
                    "repo_path": e["repo_path"],
                }
            )
        manifest = {
            "annotator": annotator,
            "proposal": proposal_id,
            "created_at": now_timestamps()["utc"],
            "blind": blind,
            "episodes": manifest_entries,
        }
        zf.writestr(f"{annotator}/{MANIFEST_NAME}", json.dumps(manifest, indent=2))
    print(f"wrote {zip_path} ({len(manifest_entries)} episodes, {len(missing)} missing)", file=sys.stderr)
    return zip_path, missing
