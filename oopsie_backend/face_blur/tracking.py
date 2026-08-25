"""Conservative temporal face tracking and blur-region planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from oopsie_backend.face_blur.types import FaceBox, FaceDetection


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Controls association, temporal gap filling, and mask padding."""

    max_gap_frames: int = 8
    hold_frames: int = 8
    pre_roll_frames: int = 2
    smoothing_window: int = 2
    min_iou: float = 0.1
    max_center_distance: float = 1.75
    max_size_ratio: float = 3.0
    motion_decay: float = 0.9
    hold_expansion_per_frame: float = 0.04
    padding_horizontal: float = 0.25
    padding_top: float = 0.35
    padding_bottom: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "max_gap_frames",
            "hold_frames",
            "pre_roll_frames",
            "smoothing_window",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.min_iou <= 1:
            raise ValueError("min_iou must be between zero and one")
        if self.max_center_distance <= 0:
            raise ValueError("max_center_distance must be positive")
        if self.max_size_ratio < 1:
            raise ValueError("max_size_ratio must be at least one")
        if not 0 < self.motion_decay <= 1:
            raise ValueError("motion_decay must be in (0, 1]")
        for name in (
            "hold_expansion_per_frame",
            "padding_horizontal",
            "padding_top",
            "padding_bottom",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackConfirmationConfig:
    """Use strong detections to confirm tracks and weaker ones to continue them."""

    continuation_threshold: float = 0.2
    seed_threshold: float = 0.3
    min_seed_observations: int = 2
    render_unconfirmed_seeds: bool = True
    fallback_pre_roll_frames: int = 2
    fallback_hold_frames: int = 15

    def __post_init__(self) -> None:
        if not 0 <= self.continuation_threshold <= 1:
            raise ValueError("continuation_threshold must be between zero and one")
        if not 0 <= self.seed_threshold <= 1:
            raise ValueError("seed_threshold must be between zero and one")
        if self.continuation_threshold > self.seed_threshold:
            raise ValueError("continuation_threshold cannot exceed seed_threshold")
        if self.min_seed_observations <= 0:
            raise ValueError("min_seed_observations must be positive")
        if self.fallback_pre_roll_frames < 0:
            raise ValueError("fallback_pre_roll_frames must be non-negative")
        if self.fallback_hold_frames < 0:
            raise ValueError("fallback_hold_frames must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_index: int
    box: FaceBox
    score: float

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "bbox": self.box.to_list(),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True, slots=True)
class FaceTrack:
    track_id: int
    observations: tuple[TrackObservation, ...]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "first_detection_frame": self.observations[0].frame_index,
            "last_detection_frame": self.observations[-1].frame_index,
            "detection_count": len(self.observations),
            "max_score": round(max(item.score for item in self.observations), 4),
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass(frozen=True, slots=True)
class TrackedFace:
    track_id: int
    box: FaceBox
    source: str

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.box.to_list(),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TrackingPlan:
    tracks: tuple[FaceTrack, ...]
    faces_by_frame: dict[int, tuple[TrackedFace, ...]]
    frame_count: int
    source_counts: dict[str, int]
    policy: dict[str, object] = field(default_factory=dict)

    @property
    def masked_frame_count(self) -> int:
        return len(self.faces_by_frame)

    @property
    def region_count(self) -> int:
        return sum(len(faces) for faces in self.faces_by_frame.values())

    def to_summary_dict(self) -> dict:
        result = {
            "track_count": len(self.tracks),
            "masked_frame_count": self.masked_frame_count,
            "region_count": self.region_count,
            "source_counts": dict(sorted(self.source_counts.items())),
            "tracks": [track.to_dict() for track in self.tracks],
        }
        if self.policy:
            result["policy"] = self.policy
        return result


@dataclass(slots=True)
class _MutableTrack:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)


def box_iou(first: FaceBox, second: FaceBox) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(first: FaceBox, second: FaceBox) -> float:
    first_x, first_y = first.center
    second_x, second_y = second.center
    average_width = (first.width + second.width) / 2
    average_height = (first.height + second.height) / 2
    scale = max(1.0, math.hypot(average_width, average_height))
    return math.hypot(first_x - second_x, first_y - second_y) / scale


def _size_ratio(first: FaceBox, second: FaceBox) -> float:
    width_ratio = max(first.width, second.width) / max(
        1e-6, min(first.width, second.width)
    )
    height_ratio = max(first.height, second.height) / max(
        1e-6, min(first.height, second.height)
    )
    return max(width_ratio, height_ratio)


def _box_velocity(
    first: FaceBox,
    first_frame: int,
    second: FaceBox,
    second_frame: int,
) -> tuple[float, float, float, float]:
    elapsed = max(1, second_frame - first_frame)
    first_x, first_y = first.center
    second_x, second_y = second.center
    return (
        (second_x - first_x) / elapsed,
        (second_y - first_y) / elapsed,
        (second.width - first.width) / elapsed,
        (second.height - first.height) / elapsed,
    )


def _project_box(
    reference: FaceBox,
    velocity: tuple[float, float, float, float],
    amount: float,
) -> FaceBox:
    center_x, center_y = reference.center
    velocity_x, velocity_y, velocity_width, velocity_height = velocity
    width = min(
        reference.width * 2,
        max(reference.width * 0.5, reference.width + velocity_width * amount),
    )
    height = min(
        reference.height * 2,
        max(reference.height * 0.5, reference.height + velocity_height * amount),
    )
    center_x += velocity_x * amount
    center_y += velocity_y * amount
    return FaceBox(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def _predict_track(track: _MutableTrack, frame_index: int) -> FaceBox:
    latest = track.observations[-1]
    if len(track.observations) < 2:
        return latest.box
    previous = track.observations[-2]
    velocity = _box_velocity(
        previous.box,
        previous.frame_index,
        latest.box,
        latest.frame_index,
    )
    return _project_box(latest.box, velocity, frame_index - latest.frame_index)


def associate_detections(
    detections_by_frame: Sequence[Sequence[FaceDetection]],
    config: TrackingConfig,
) -> tuple[FaceTrack, ...]:
    """Associate per-frame detections into deterministic, short-gap tracks."""
    tracks: list[_MutableTrack] = []
    next_track_id = 0
    for frame_index, detections in enumerate(detections_by_frame):
        valid_detections = [item for item in detections if item.area > 0]
        active = [
            track
            for track in tracks
            if frame_index - track.observations[-1].frame_index - 1
            <= config.max_gap_frames
        ]
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active):
            predicted = _predict_track(track, frame_index)
            missing_frames = frame_index - track.observations[-1].frame_index - 1
            for detection_index, detection in enumerate(valid_detections):
                overlap = box_iou(predicted, detection.box)
                distance = _center_distance(predicted, detection.box)
                if overlap < config.min_iou and distance > config.max_center_distance:
                    continue
                if _size_ratio(predicted, detection.box) > config.max_size_ratio:
                    continue
                quality = overlap * 2 + 1 / (1 + distance) + 0.05 / (1 + missing_frames)
                candidates.append((quality, track_index, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = valid_detections[detection_index]
            active[track_index].observations.append(
                TrackObservation(frame_index, detection.box, detection.score)
            )
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(valid_detections):
            if detection_index in assigned_detections:
                continue
            tracks.append(
                _MutableTrack(
                    track_id=next_track_id,
                    observations=[
                        TrackObservation(frame_index, detection.box, detection.score)
                    ],
                )
            )
            next_track_id += 1

    return tuple(
        FaceTrack(track.track_id, tuple(track.observations)) for track in tracks
    )


def _smoothed_anchor_boxes(
    track: FaceTrack,
    smoothing_window: int,
) -> dict[int, FaceBox]:
    anchors: dict[int, FaceBox] = {}
    for observation in track.observations:
        coordinates = [0.0, 0.0, 0.0, 0.0]
        total_weight = 0.0
        for neighbor in track.observations:
            distance = abs(neighbor.frame_index - observation.frame_index)
            if distance > smoothing_window:
                continue
            weight = float(smoothing_window + 1 - distance)
            for index, value in enumerate(neighbor.box.to_list(digits=12)):
                coordinates[index] += value * weight
            total_weight += weight
        averaged = FaceBox(*(value / total_weight for value in coordinates))
        # Never let smoothing contract away a detector-supported face pixel.
        anchors[observation.frame_index] = observation.box.union(averaged)
    return anchors


def _damped_distance(step: int, decay: float) -> float:
    if decay == 1:
        return float(step)
    return (1 - decay**step) / (1 - decay)


def _plan_tracks(
    tracks: Sequence[FaceTrack],
    *,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    config: TrackingConfig,
    fallback_tracks: Sequence[FaceTrack] = (),
    fallback_pre_roll_frames: int = 0,
    fallback_hold_frames: int = 0,
    reported_tracks: Sequence[FaceTrack] | None = None,
    policy: dict[str, object] | None = None,
) -> TrackingPlan:
    planned: dict[int, list[TrackedFace]] = {}

    def add(frame_index: int, track_id: int, box: FaceBox, source: str) -> None:
        if frame_index < 0 or frame_index >= frame_count:
            return
        padded = box.expanded(
            horizontal=config.padding_horizontal,
            top=config.padding_top,
            bottom=config.padding_bottom,
        ).clipped(frame_width, frame_height)
        if padded.area <= 0:
            return
        planned.setdefault(frame_index, []).append(
            TrackedFace(track_id=track_id, box=padded, source=source)
        )

    def plan_track(
        track: FaceTrack,
        *,
        pre_roll_frames: int,
        hold_frames: int,
        fallback: bool,
    ) -> None:
        observations = track.observations
        anchors = _smoothed_anchor_boxes(track, config.smoothing_window)
        observed_frames = {item.frame_index for item in observations}

        def source(name: str) -> str:
            return f"fallback_{name}" if fallback else name

        first = observations[0]
        first_velocity = (0.0, 0.0, 0.0, 0.0)
        if len(observations) >= 2:
            second = observations[1]
            first_velocity = _box_velocity(
                anchors[first.frame_index],
                first.frame_index,
                anchors[second.frame_index],
                second.frame_index,
            )
        for step in range(pre_roll_frames, 0, -1):
            amount = -_damped_distance(step, config.motion_decay)
            projected = _project_box(
                anchors[first.frame_index], first_velocity, amount
            ).scaled_about_center(1 + config.hold_expansion_per_frame * step)
            add(first.frame_index - step, track.track_id, projected, source("pre_roll"))

        for index, observation in enumerate(observations):
            start_frame = observation.frame_index
            start_box = anchors[start_frame]
            add(start_frame, track.track_id, start_box, source("detection"))
            if index + 1 >= len(observations):
                continue
            next_frame = observations[index + 1].frame_index
            next_box = anchors[next_frame]
            elapsed = next_frame - start_frame
            for frame_index in range(start_frame + 1, next_frame):
                region_source = (
                    "detection" if frame_index in observed_frames else "interpolated"
                )
                add(
                    frame_index,
                    track.track_id,
                    start_box.interpolate(
                        next_box, (frame_index - start_frame) / elapsed
                    ),
                    source(region_source),
                )

        last = observations[-1]
        last_velocity = (0.0, 0.0, 0.0, 0.0)
        if len(observations) >= 2:
            previous = observations[-2]
            last_velocity = _box_velocity(
                anchors[previous.frame_index],
                previous.frame_index,
                anchors[last.frame_index],
                last.frame_index,
            )
        for step in range(1, hold_frames + 1):
            amount = _damped_distance(step, config.motion_decay)
            projected = _project_box(
                anchors[last.frame_index], last_velocity, amount
            ).scaled_about_center(1 + config.hold_expansion_per_frame * step)
            add(last.frame_index + step, track.track_id, projected, source("hold"))

    for track in tracks:
        plan_track(
            track,
            pre_roll_frames=config.pre_roll_frames,
            hold_frames=config.hold_frames,
            fallback=False,
        )
    for track in fallback_tracks:
        plan_track(
            track,
            pre_roll_frames=fallback_pre_roll_frames,
            hold_frames=fallback_hold_frames,
            fallback=True,
        )

    frozen = {
        frame_index: tuple(sorted(faces, key=lambda item: item.track_id))
        for frame_index, faces in sorted(planned.items())
    }
    source_counts: dict[str, int] = {}
    for faces in frozen.values():
        for face in faces:
            source_counts[face.source] = source_counts.get(face.source, 0) + 1
    return TrackingPlan(
        tracks=tuple(tracks if reported_tracks is None else reported_tracks),
        faces_by_frame=frozen,
        frame_count=frame_count,
        source_counts=source_counts,
        policy=policy or {},
    )


def plan_blur_regions(
    detections_by_frame: Sequence[Sequence[FaceDetection]],
    *,
    frame_width: int,
    frame_height: int,
    config: TrackingConfig | None = None,
) -> TrackingPlan:
    """Build padded regions, interpolating misses and holding ended tracks."""
    tracking_config = config or TrackingConfig()
    tracks = associate_detections(detections_by_frame, tracking_config)
    return _plan_tracks(
        tracks,
        frame_count=len(detections_by_frame),
        frame_width=frame_width,
        frame_height=frame_height,
        config=tracking_config,
    )


def plan_confirmed_blur_regions(
    detections_by_frame: Sequence[Sequence[FaceDetection]],
    *,
    frame_width: int,
    frame_height: int,
    tracking: TrackingConfig | None = None,
    confirmation: TrackConfirmationConfig | None = None,
) -> TrackingPlan:
    """Plan only seeded tracks while preserving isolated strong detections.

    Detections below ``seed_threshold`` can extend a spatially associated track
    but can never create a rendered track by themselves. Tracks with enough
    strong observations receive the normal interpolation, pre-roll, and hold.
    An unconfirmed track with a strong seed retains its complete associated
    trajectory and receives a separately configured conservative temporal
    window. This avoids exposing a persistent face when the detector produces
    only one strong frame while keeping low-score-only tracks discarded.
    """
    tracking_config = tracking or TrackingConfig()
    confirmation_config = confirmation or TrackConfirmationConfig()
    candidates = tuple(
        tuple(
            detection
            for detection in detections
            if detection.area > 0
            and detection.score >= confirmation_config.continuation_threshold
        )
        for detections in detections_by_frame
    )
    candidate_tracks = associate_detections(candidates, tracking_config)
    confirmed_tracks: list[FaceTrack] = []
    fallback_tracks: list[FaceTrack] = []
    fallback_seed_count = 0
    for track in candidate_tracks:
        seed_observations = tuple(
            observation
            for observation in track.observations
            if observation.score >= confirmation_config.seed_threshold
        )
        if len(seed_observations) >= confirmation_config.min_seed_observations:
            confirmed_tracks.append(track)
            continue
        if confirmation_config.render_unconfirmed_seeds and seed_observations:
            fallback_tracks.append(track)
            fallback_seed_count += len(seed_observations)

    rendered_tracks = tuple(
        sorted((*confirmed_tracks, *fallback_tracks), key=lambda item: item.track_id)
    )
    return _plan_tracks(
        confirmed_tracks,
        frame_count=len(detections_by_frame),
        frame_width=frame_width,
        frame_height=frame_height,
        config=tracking_config,
        fallback_tracks=fallback_tracks,
        fallback_pre_roll_frames=confirmation_config.fallback_pre_roll_frames,
        fallback_hold_frames=confirmation_config.fallback_hold_frames,
        reported_tracks=rendered_tracks,
        policy={
            "mode": "confirmed_hysteresis",
            **confirmation_config.to_dict(),
            "candidate_track_count": len(candidate_tracks),
            "confirmed_track_count": len(confirmed_tracks),
            "fallback_track_count": len(fallback_tracks),
            "fallback_detection_count": fallback_seed_count,
            "fallback_observation_count": sum(
                len(track.observations) for track in fallback_tracks
            ),
            "discarded_track_count": (
                len(candidate_tracks) - len(confirmed_tracks) - len(fallback_tracks)
            ),
        },
    )


__all__ = [
    "FaceTrack",
    "TrackConfirmationConfig",
    "TrackedFace",
    "TrackingConfig",
    "TrackingPlan",
    "associate_detections",
    "box_iou",
    "plan_blur_regions",
    "plan_confirmed_blur_regions",
]
