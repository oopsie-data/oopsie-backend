"""Detector adapters with a common source-frame coordinate contract."""

from __future__ import annotations

from pathlib import Path

from oopsie_backend.face_blur.detectors.base import FaceDetector


def create_detector(
    name: str,
    model_path: Path,
    *,
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    device: str,
) -> FaceDetector:
    """Create a configured detector without importing unused runtimes."""
    if name == "yunet":
        from oopsie_backend.face_blur.detectors.yunet import YuNetDetector

        return YuNetDetector(
            model_path,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
    if name == "deface-centerface":
        from oopsie_backend.face_blur.detectors.deface import (
            DefaceCenterFaceDetector,
        )

        return DefaceCenterFaceDetector(
            model_path,
            score_threshold=score_threshold,
            device=device,
        )
    if name.startswith("scrfd-"):
        from oopsie_backend.face_blur.detectors.scrfd import ScrfdDetector

        return ScrfdDetector(
            model_path,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            device=device,
        )
    raise ValueError(f"unknown detector: {name!r}")


__all__ = ["FaceDetector", "create_detector"]
