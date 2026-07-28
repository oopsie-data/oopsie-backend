"""Adapter for the CenterFace detector bundled with ORB-HD/deface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oopsie_backend.face_blur.detectors.base import FaceDetector
from oopsie_backend.face_blur.types import FaceDetection


def execution_provider_for_device(device: str, available: list[str]) -> str:
    """Resolve the project's device syntax to a deface ONNX provider name."""
    normalized = device.strip().lower()
    if normalized == "auto":
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"
    if normalized == "cpu":
        provider = "CPUExecutionProvider"
    elif normalized in {"cuda", "cuda:0"}:
        provider = "CUDAExecutionProvider"
    elif normalized.startswith("cuda:"):
        raise ValueError(
            "deface only accepts cuda:0 through its public provider API; select "
            "another GPU with CUDA_VISIBLE_DEVICES"
        )
    else:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:0'")
    if provider not in available:
        raise RuntimeError(
            f"deface requested {provider}, but available providers are {available}"
        )
    return provider


class DefaceCenterFaceDetector(FaceDetector):
    """Expose deface's native-resolution CenterFace detections to our benchmark."""

    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = 0.2,
        device: str = "cpu",
    ) -> None:
        try:
            import deface
            import onnxruntime as ort
            from deface.centerface import CenterFace
        except ImportError as exc:
            raise RuntimeError(
                "deface CenterFace requires the face-blur project extra"
            ) from exc

        self.model_path = model_path.resolve()
        self.score_threshold = score_threshold
        provider = execution_provider_for_device(
            device, list(ort.get_available_providers())
        )
        if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        self.centerface = CenterFace(
            onnx_path=str(self.model_path),
            in_shape=None,
            backend="onnxrt",
            override_execution_provider=provider,
        )
        actual_providers = list(self.centerface.sess.get_providers())
        if not actual_providers or actual_providers[0] != provider:
            raise RuntimeError(
                f"deface did not activate {provider} (active: {actual_providers})"
            )
        self._metadata = {
            "adapter": "deface-centerface",
            "package": "deface",
            "package_version": deface.__version__,
            "model_path": str(self.model_path),
            "input_size": None,
            "resolution_mode": "native",
            "score_threshold": score_threshold,
            "nms_threshold": 0.3,
            "requested_device": device,
            "providers": actual_providers,
            "onnxruntime_version": ort.__version__,
            "uses_landmarks": True,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("deface expects a BGR image with shape HxWx3")
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, landmarks = self.centerface(frame_rgb, threshold=self.score_threshold)
        detections: list[FaceDetection] = []
        for index, row in enumerate(boxes):
            points: tuple[tuple[float, float], ...] = ()
            if index < len(landmarks):
                values = landmarks[index]
                points = tuple(
                    (float(values[offset]), float(values[offset + 1]))
                    for offset in range(0, len(values), 2)
                )
            detection = FaceDetection(
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                score=float(row[4]),
                landmarks=points,
            ).clipped(frame_width, frame_height)
            if detection.area > 0:
                detections.append(detection)
        return detections


__all__ = ["DefaceCenterFaceDetector", "execution_provider_for_device"]
