"""InsightFace SCRFD ONNX detector adapter.

The output decoder is adapted from InsightFace's MIT-licensed
``python-package/insightface/model_zoo/retinaface.py`` at tag ``v0.7``.
Only face detection is implemented; no recognition or embeddings are loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oopsie_backend.face_blur.detectors.base import FaceDetector
from oopsie_backend.face_blur.types import FaceDetection


def distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode left/top/right/bottom distances around anchor centers."""
    return np.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        axis=-1,
    )


def distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    coordinates: list[np.ndarray] = []
    for index in range(0, distances.shape[1], 2):
        coordinates.append(points[:, 0] + distances[:, index])
        coordinates.append(points[:, 1] + distances[:, index + 1])
    return np.stack(coordinates, axis=-1)


def non_max_suppression(boxes: np.ndarray, threshold: float) -> list[int]:
    """Return score-descending indices kept after IoU NMS."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = (boxes[:, index] for index in range(5))
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        others = order[1:]
        if not others.size:
            break
        xx1 = np.maximum(x1[current], x1[others])
        yy1 = np.maximum(y1[current], y1[others])
        xx2 = np.minimum(x2[current], x2[others])
        yy2 = np.minimum(y2[current], y2[others])
        intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        union = areas[current] + areas[others] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = others[np.where(iou <= threshold)[0]]
    return keep


def _providers_for_device(device: str) -> list[Any]:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        device_id = 0
    elif normalized.startswith("cuda:"):
        try:
            device_id = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device: {device!r}") from exc
    else:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    return [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]


class ScrfdDetector(FaceDetector):
    """SCRFD inference through ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.1,
        nms_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "SCRFD requires ONNX Runtime; install the face-blur or "
                "face-blur-gpu project extra"
            ) from exc

        self.model_path = model_path.resolve()
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        requested_providers = _providers_for_device(device)
        # ONNX Runtime's ``[cuda,cudnn]`` extra installs NVIDIA libraries as
        # Python packages. Preload them before creating the execution provider
        # so users do not need to construct an LD_LIBRARY_PATH manually.
        if device.lower().startswith("cuda") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=requested_providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not initialize SCRFD on {device!r}: {exc}"
            ) from exc

        actual_providers = self.session.get_providers()
        if device.lower().startswith("cuda") and (
            not actual_providers or actual_providers[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "CUDA was requested but ONNX Runtime did not activate the CUDA "
                f"provider (active providers: {actual_providers})"
            )

        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        output_count = len(self.output_names)
        if output_count in (6, 9):
            self.feature_strides = (8, 16, 32)
            self.num_anchors = 2
        elif output_count in (10, 15):
            self.feature_strides = (8, 16, 32, 64, 128)
            self.num_anchors = 1
        else:
            raise RuntimeError(
                f"unsupported SCRFD output count {output_count} in {model_path}"
            )
        self.use_landmarks = output_count in (9, 15)
        self.feature_map_count = len(self.feature_strides)
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._metadata = {
            "adapter": "scrfd",
            "model_path": str(self.model_path),
            "input_size": list(input_size),
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "requested_device": device,
            "providers": actual_providers,
            "onnxruntime_version": ort.__version__,
            "output_names": self.output_names,
            "feature_strides": list(self.feature_strides),
            "uses_landmarks": self.use_landmarks,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self.center_cache.get(key)
        if cached is not None:
            return cached
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
        if len(self.center_cache) < 100:
            self.center_cache[key] = centers
        return centers

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("SCRFD expects a BGR image with shape HxWx3")
        frame_height, frame_width = frame_bgr.shape[:2]
        target_width, target_height = self.input_size
        image_ratio = frame_height / frame_width
        model_ratio = target_height / target_width
        if image_ratio > model_ratio:
            resized_height = target_height
            resized_width = max(1, int(resized_height / image_ratio))
        else:
            resized_width = target_width
            resized_height = max(1, int(resized_width * image_ratio))
        scale = resized_height / frame_height
        resized = cv2.resize(frame_bgr, (resized_width, resized_height))
        detector_input = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        detector_input[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            detector_input,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(
            self.output_names,
            {self.input_name: blob},
        )

        scores_by_level: list[np.ndarray] = []
        boxes_by_level: list[np.ndarray] = []
        landmarks_by_level: list[np.ndarray] = []
        for level, stride in enumerate(self.feature_strides):
            scores = np.asarray(outputs[level]).reshape(-1)
            distances = np.asarray(outputs[level + self.feature_map_count])
            distances = distances.reshape((-1, 4)) * stride
            feature_height = target_height // stride
            feature_width = target_width // stride
            centers = self._anchor_centers(feature_height, feature_width, stride)
            if (
                scores.shape[0] != centers.shape[0]
                or distances.shape[0] != centers.shape[0]
            ):
                raise RuntimeError(
                    "SCRFD output shape does not match configured input size: "
                    f"scores={scores.shape}, boxes={distances.shape}, "
                    f"centers={centers.shape}"
                )
            positive = np.where(scores >= self.score_threshold)[0]
            if not positive.size:
                continue
            decoded_boxes = distance_to_bbox(centers, distances)
            scores_by_level.append(scores[positive])
            boxes_by_level.append(decoded_boxes[positive])
            if self.use_landmarks:
                landmark_distances = np.asarray(
                    outputs[level + self.feature_map_count * 2]
                ).reshape((-1, 10))
                decoded_landmarks = distance_to_landmarks(
                    centers, landmark_distances * stride
                ).reshape((-1, 5, 2))
                landmarks_by_level.append(decoded_landmarks[positive])

        if not scores_by_level:
            return []
        scores = np.concatenate(scores_by_level)
        boxes = np.vstack(boxes_by_level) / scale
        order = scores.argsort()[::-1]
        candidates = np.column_stack([boxes, scores]).astype(np.float32)[order]
        kept = non_max_suppression(candidates, self.nms_threshold)
        candidates = candidates[kept]
        landmarks = None
        if self.use_landmarks:
            landmarks = (np.vstack(landmarks_by_level) / scale)[order][kept]

        detections: list[FaceDetection] = []
        for index, row in enumerate(candidates):
            points: tuple[tuple[float, float], ...] = ()
            if landmarks is not None:
                points = tuple((float(x), float(y)) for x, y in landmarks[index])
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
