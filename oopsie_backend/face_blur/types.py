"""Shared data structures for face detectors and downstream blurring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FaceBox:
    """A face region in source-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, frame_width: int, frame_height: int) -> FaceBox:
        """Return a copy clamped to a frame's valid coordinate range."""
        return FaceBox(
            x1=min(max(self.x1, 0.0), float(frame_width)),
            y1=min(max(self.y1, 0.0), float(frame_height)),
            x2=min(max(self.x2, 0.0), float(frame_width)),
            y2=min(max(self.y2, 0.0), float(frame_height)),
        )

    def union(self, other: FaceBox) -> FaceBox:
        """Return the smallest box containing both boxes."""
        return FaceBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def interpolate(self, other: FaceBox, amount: float) -> FaceBox:
        """Linearly interpolate this box toward *other*."""
        return FaceBox(
            self.x1 + (other.x1 - self.x1) * amount,
            self.y1 + (other.y1 - self.y1) * amount,
            self.x2 + (other.x2 - self.x2) * amount,
            self.y2 + (other.y2 - self.y2) * amount,
        )

    def expanded(
        self,
        *,
        horizontal: float,
        top: float,
        bottom: float,
    ) -> FaceBox:
        """Expand by fractions of the current width and height."""
        return FaceBox(
            self.x1 - self.width * horizontal,
            self.y1 - self.height * top,
            self.x2 + self.width * horizontal,
            self.y2 + self.height * bottom,
        )

    def translated(self, dx: float, dy: float) -> FaceBox:
        return FaceBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def scaled_about_center(self, factor: float) -> FaceBox:
        center_x, center_y = self.center
        half_width = self.width * factor / 2
        half_height = self.height * factor / 2
        return FaceBox(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )

    def to_list(self, *, digits: int = 4) -> list[float]:
        return [
            round(self.x1, digits),
            round(self.y1, digits),
            round(self.x2, digits),
            round(self.y2, digits),
        ]


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One face bounding box in source-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def box(self) -> FaceBox:
        return FaceBox(self.x1, self.y1, self.x2, self.y2)

    def clipped(self, frame_width: int, frame_height: int) -> FaceDetection:
        """Return a copy clamped to a frame's valid coordinate range."""
        return FaceDetection(
            x1=min(max(self.x1, 0.0), float(frame_width)),
            y1=min(max(self.y1, 0.0), float(frame_height)),
            x2=min(max(self.x2, 0.0), float(frame_width)),
            y2=min(max(self.y2, 0.0), float(frame_height)),
            score=min(max(self.score, 0.0), 1.0),
            landmarks=tuple(
                (
                    min(max(x, 0.0), float(frame_width)),
                    min(max(y, 0.0), float(frame_height)),
                )
                for x, y in self.landmarks
            ),
        )

    def to_dict(self, *, digits: int = 4) -> dict:
        """Return a compact JSON-serializable representation."""
        result = {
            "bbox": [
                round(self.x1, digits),
                round(self.y1, digits),
                round(self.x2, digits),
                round(self.y2, digits),
            ],
            "score": round(self.score, digits),
        }
        if self.landmarks:
            result["landmarks"] = [
                [round(x, digits), round(y, digits)] for x, y in self.landmarks
            ]
        return result
