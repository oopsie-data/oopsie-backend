"""Abstract face-detector interface used by evaluation and blurring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from oopsie_backend.face_blur.types import FaceDetection


class FaceDetector(ABC):
    """A detector returning boxes in the input frame's coordinate system."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces in one uint8 BGR frame."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable runtime and model metadata."""
