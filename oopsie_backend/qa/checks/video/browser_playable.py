"""Checks whether every camera video uses a codec and pixel format that web browsers play."""

from __future__ import annotations

import re
import subprocess

import imageio_ffmpeg

from oopsie_backend.qa.action_qa import _finding, _skip
from oopsie_backend.qa.checks.base import VIDEO, Check, check
from oopsie_backend.qa.video_checks import ALL_CAMERAS, CATEGORY, drop_ignored_cameras

STREAM = re.compile(r"Video: (\w+)[^,]*, (\w+)")


def video_format(path) -> tuple[str, str] | None:
    """``(codec, pixel format)`` of the first video stream, from ``ffmpeg -i``."""
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False
    )
    m = STREAM.search(proc.stderr)
    return (m.group(1), m.group(2)) if m else None


@check("video", "browser_playable", VIDEO)
def browser_playable(ctx, config: dict, spec: Check) -> dict:
    cfg = config["video"]["browser_playable"]
    if not cfg["enabled"]:
        return {"findings": [], "skips": [], "checked": []}
    videos, ignored = drop_ignored_cameras(ctx.videos, config["video_settings"]["ignore_cameras"])
    skips = [_skip(CATEGORY, spec.test, cam, "ignored_camera") for cam in ignored]
    if not videos:
        skips.append(_skip(CATEGORY, spec.test, ALL_CAMERAS, "no_video_files"))
        return {"findings": [], "skips": skips, "checked": []}
    findings = []
    for camera, path in sorted(videos.items()):
        fmt = video_format(path)
        if fmt is None:
            skips.append(_skip(CATEGORY, spec.test, camera, "no_video_stream"))
            continue
        codec, pix_fmt = fmt
        if codec not in cfg["codecs"] or pix_fmt not in cfg["pixel_formats"]:
            issue = f"{codec} {pix_fmt} does not play in web browsers"
            findings.append(_finding(CATEGORY, spec.test, camera, issue=issue, codec=codec, pixel_format=pix_fmt))
    return {"findings": findings, "skips": skips, "checked": [(CATEGORY, spec.test)]}
