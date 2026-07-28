"""OpenCV YuNet face detector adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oopsie_backend.face_blur.detectors.base import FaceDetector
from oopsie_backend.face_blur.types import FaceDetection


class YuNetDetector(FaceDetector):
    """YuNet with letterboxing and source-coordinate output."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.1,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = model_path.resolve()
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._metadata = {
            "adapter": "yunet",
            "model_path": str(self.model_path),
            "input_size": list(input_size),
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "top_k": top_k,
            "requested_device": "cpu",
            "providers": ["OpenCV-DNN-CPU"],
            "opencv_version": cv2.__version__,
            "uses_landmarks": True,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("YuNet expects a BGR image with shape HxWx3")
        frame_height, frame_width = frame_bgr.shape[:2]
        target_width, target_height = self.input_size
        scale = min(target_width / frame_width, target_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame_bgr, (resized_width, resized_height))
        detector_input = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        detector_input[:resized_height, :resized_width] = resized
        _, faces = self.detector.detect(detector_input)
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for face in faces:
            x, y, width, height = (float(value) / scale for value in face[:4])
            landmarks = tuple(
                (float(face[index]) / scale, float(face[index + 1]) / scale)
                for index in range(4, 14, 2)
            )
            detection = FaceDetection(
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                score=float(face[14]),
                landmarks=landmarks,
            ).clipped(frame_width, frame_height)
            if detection.area > 0:
                detections.append(detection)
        return detections
