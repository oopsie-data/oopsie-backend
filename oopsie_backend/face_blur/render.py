"""Render tracked face regions onto decoded video frames."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from oopsie_backend.face_blur.tracking import TrackedFace


@dataclass(frozen=True, slots=True)
class BlurConfig:
    """Controls the irreversible visual anonymization applied to each region."""

    method: str = "gaussian"
    gaussian_kernel_fraction: float = 0.5
    gaussian_min_kernel: int = 31
    gaussian_passes: int = 2
    pixel_block_size: int = 12

    def __post_init__(self) -> None:
        if self.method not in {"gaussian", "pixelate"}:
            raise ValueError("method must be 'gaussian' or 'pixelate'")
        if self.gaussian_kernel_fraction <= 0:
            raise ValueError("gaussian_kernel_fraction must be positive")
        if self.gaussian_min_kernel <= 0:
            raise ValueError("gaussian_min_kernel must be positive")
        if self.gaussian_passes <= 0:
            raise ValueError("gaussian_passes must be positive")
        if self.pixel_block_size <= 1:
            raise ValueError("pixel_block_size must be greater than one")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DefaceMaskConfig:
    """Visual settings compatible with deface's native anonymizer."""

    mask_scale: float = 1.3
    ellipse: bool = True
    replacewith: str = "blur"
    mosaic_size: int = 20

    def __post_init__(self) -> None:
        if self.mask_scale <= 0:
            raise ValueError("mask_scale must be positive")
        if self.replacewith not in {"blur", "solid", "mosaic"}:
            raise ValueError("unsupported replacement method")
        if self.mosaic_size <= 0:
            raise ValueError("mosaic_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def _odd_kernel(value: float, minimum: int) -> int:
    kernel = max(minimum, int(round(value)))
    return kernel if kernel % 2 else kernel + 1


def blur_frame(
    frame_bgr: np.ndarray,
    faces: Sequence[TrackedFace],
    config: BlurConfig | None = None,
) -> np.ndarray:
    """Return a copy with every tracked face region aggressively obscured."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("blur_frame expects a BGR image with shape HxWx3")
    blur_config = config or BlurConfig()
    output = frame_bgr.copy()
    frame_height, frame_width = output.shape[:2]
    for face in faces:
        clipped = face.box.clipped(frame_width, frame_height)
        x1 = max(0, int(math.floor(clipped.x1)))
        y1 = max(0, int(math.floor(clipped.y1)))
        x2 = min(frame_width, int(math.ceil(clipped.x2)))
        y2 = min(frame_height, int(math.ceil(clipped.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        region = output[y1:y2, x1:x2]
        if blur_config.method == "gaussian":
            kernel = _odd_kernel(
                min(region.shape[:2]) * blur_config.gaussian_kernel_fraction,
                blur_config.gaussian_min_kernel,
            )
            obscured = region
            for _ in range(blur_config.gaussian_passes):
                obscured = cv2.GaussianBlur(
                    obscured,
                    (kernel, kernel),
                    sigmaX=0,
                    borderType=cv2.BORDER_REPLICATE,
                )
        else:
            down_width = max(
                1, math.ceil(region.shape[1] / blur_config.pixel_block_size)
            )
            down_height = max(
                1, math.ceil(region.shape[0] / blur_config.pixel_block_size)
            )
            tiny = cv2.resize(
                region,
                (down_width, down_height),
                interpolation=cv2.INTER_AREA,
            )
            obscured = cv2.resize(
                tiny,
                (region.shape[1], region.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        output[y1:y2, x1:x2] = obscured
    return output


def blur_frame_deface(
    frame_bgr: np.ndarray,
    faces: Sequence[TrackedFace],
    config: DefaceMaskConfig | None = None,
) -> np.ndarray:
    """Apply deface's own mask geometry and anonymizer to planned regions."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("blur_frame_deface expects a BGR image with shape HxWx3")
    from deface.deface import anonymize_frame

    mask_config = config or DefaceMaskConfig()
    output = frame_bgr.copy()
    detections = np.asarray(
        [
            [face.box.x1, face.box.y1, face.box.x2, face.box.y2, 1.0]
            for face in faces
            if face.box.area > 0
        ],
        dtype=np.float32,
    )
    if detections.size:
        anonymize_frame(
            detections,
            output,
            mask_scale=mask_config.mask_scale,
            replacewith=mask_config.replacewith,
            ellipse=mask_config.ellipse,
            draw_scores=False,
            replaceimg=None,
            mosaicsize=mask_config.mosaic_size,
        )
    return output


__all__ = ["BlurConfig", "DefaceMaskConfig", "blur_frame", "blur_frame_deface"]
