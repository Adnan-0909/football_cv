"""
Player Tracking Module
======================

Associates detections across consecutive frames so every player keeps a stable
ID. Two ByteTrack implementations sit behind the single :class:`PlayerTracker`
interface, selected with ``tracker_type`` in ``config.yaml``:

``bytetrack`` (default)
    Ultralytics' battle-tested ``BYTETracker``: Kalman-filtered motion plus the
    two-stage ByteTrack association (high-confidence detections first, then a
    low-confidence rescue pass). Preferred because it is maintained upstream
    and copes better with occlusions. Requires the ``lap`` package, which is
    listed in ``requirements.txt``.

``bytetrack_lite``
    A dependency-free NumPy port of the same idea, kept as a fallback for
    machines where ``lap`` cannot be installed. Matching uses
    :mod:`app.matching` (pure NumPy), motion uses a smoothed constant-velocity
    model instead of a Kalman filter.

Both backends behave identically for callers:

1. Detections are split into high- and low-confidence groups.
2. Stage 1 matches *all* live tracks against high-confidence detections using
   IoU fused with the detection score.
3. Stage 2 matches leftover tracks against low-confidence detections using
   plain IoU - rescuing players that are briefly occluded or blurry.
4. Unmatched detections above ``new_track_thresh`` start new tracks; unmatched
   tracks stay alive (predicted forward) for ``track_buffer`` frames.

``update()`` returns only the tracks matched in the current frame (standard
ByteTrack behaviour), so briefly lost objects are simply not drawn until they
reappear - they are never re-labelled in the meantime.

Note: the Ultralytics backend confirms a brand-new track on its *second* hit
(every frame except the very first), which is the usual ByteTrack guard against
one-off false positives - a fresh ID therefore appears one frame after the
player is first detected. ``bytetrack_lite`` confirms immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import TrackerConfig
from app.detector import Detection
from app.matching import linear_assignment

logger = logging.getLogger(__name__)

# ByteTrack paper: second association uses raw IoU with a fixed 0.5 cost limit.
STAGE2_MATCH_THRESH = 0.5
# Smoothing factor for the constant-velocity estimate (0 = keep, 1 = instant).
VELOCITY_SMOOTHING = 0.5
# Velocity shrink applied to tracks that were not matched this frame.
VELOCITY_DECAY = 0.9
# Cross-class pairs get this cost: above both stage thresholds, so never matched.
CLASS_MISMATCH_COST = 1.0
# COCO ids used by the detector, as a fallback when a class name is unknown.
DEFAULT_CLASS_NAMES = {0: "player", 32: "ball"}

# Backend names accepted in config.yaml tracker.tracker_type.
BYTETRACK_BACKEND = "bytetrack"
BYTETRACK_LITE_BACKEND = "bytetrack_lite"


@dataclass
class TrackedObject:
    """
    Represents an object tracked across multiple frames.

    Attributes:
        track_id: Unique integer ID persisting across frames.
        bbox: Current bounding box (x1, y1, x2, y2).
        class_id: Numerical class identifier.
        class_name: Human readable class name (e.g. "player", "ball").
        trajectory: List of past centroid positions (x, y) for movement trails.
        confidence: Detection confidence of the most recent association.
    """
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class _Track:
    """Internal mutable track state (ID, motion estimate, bookkeeping)."""
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    score: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    time_since_update: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class _Observation:
    """One association returned by a backend for the current frame."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _absorb(
    track: _Track,
    bbox: Sequence[float],
    confidence: float,
    class_id: int,
    class_name: str,
    trajectory_length: int,
) -> None:
    """Fuse a matched observation into the track state."""
    new_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    new_center = _center(new_bbox)
    old_center = _center(track.bbox)

    instant = (new_center[0] - old_center[0], new_center[1] - old_center[1])
    a = VELOCITY_SMOOTHING if track.hits > 1 else 1.0
    track.velocity = (
        (1 - a) * track.velocity[0] + a * instant[0],
        (1 - a) * track.velocity[1] + a * instant[1],
    )

    track.bbox = new_bbox
    track.class_id = class_id
    track.class_name = class_name
    track.score = float(confidence)
    track.hits += 1
    track.time_since_update = 0
    track.trajectory.append(new_center)
    if len(track.trajectory) > trajectory_length:
        del track.trajectory[0 : len(track.trajectory) - trajectory_length]


def _age_track(track: _Track) -> None:
    """Age a track that was not matched this frame and damp its velocity."""
    track.time_since_update += 1
    track.velocity = (track.velocity[0] * VELOCITY_DECAY, track.velocity[1] * VELOCITY_DECAY)


def _make_track(track_id: int, observation: _Observation) -> _Track:
    """Create a brand new track from an observation."""
    return _Track(
        track_id=track_id,
        bbox=tuple(float(v) for v in observation.bbox),  # type: ignore[arg-type]
        class_id=observation.class_id,
        class_name=observation.class_name,
        score=observation.confidence,
        trajectory=[_center(observation.bbox)],
    )


def _to_tracked_object(track: _Track) -> TrackedObject:
    return TrackedObject(
        track_id=track.track_id,
        bbox=track.bbox,
        class_id=track.class_id,
        class_name=track.class_name,
        trajectory=list(track.trajectory),
        confidence=track.score,
    )


def _iou_matrix(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise IoU matrix of shape (len(boxes_a), len(boxes_b))."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class _UltralyticsByteTrack:
    """
    Adapter around Ultralytics' built-in :class:`BYTETracker`.

    The upstream tracker consumes a ``Boxes`` object (the very same thing
    ``model.track()`` hands it) and returns rows of
    ``[x1, y1, x2, y2, track_id, score, class, det_index]``. This class converts
    :class:`~app.detector.Detection` objects to that format and translates the
    result back into the shared :class:`_Track` bookkeeping, so trajectory and
    ``live_tracks`` behave exactly like the lite backend.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: List[_Track] = []
        self.frame_index = 0
        self._frame_shape: Tuple[int, int] = (1080, 1920)
        self._tracker, self._Boxes = self._build_backend()
        logger.debug("Ultralytics BYTETracker backend ready")

    # ------------------------------------------------------------------ #
    # Backend setup
    # ------------------------------------------------------------------ #

    def _build_backend(self):
        """Import BYTETracker and configure it from our TrackerConfig."""
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
        except Exception as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Ultralytics' built-in ByteTrack backend could not be imported "
                f"({type(exc).__name__}: {exc}). Install its dependency with "
                "'pip install -r requirements.txt' (needs lap>=0.5.12), or switch "
                "to the dependency-free fallback by setting "
                "tracker.tracker_type: bytetrack_lite in config.yaml."
            ) from exc

        # Start from upstream's bytetrack.yaml defaults, then apply the exact
        # thresholds documented in this project's config.yaml.
        args = IterableSimpleNamespace(**_load_bytetrack_args())
        args.track_high_thresh = float(self.config.track_high_thresh)
        args.track_low_thresh = float(self.config.track_low_thresh)
        args.new_track_thresh = float(self.config.new_track_thresh)
        args.track_buffer = int(self.config.track_buffer)
        args.match_thresh = float(self.config.match_thresh)
        args.fuse_score = bool(getattr(args, "fuse_score", True))
        return BYTETracker(args), Boxes

    # ------------------------------------------------------------------ #
    # Public backend API
    # ------------------------------------------------------------------ #

    @property
    def live_tracks(self) -> List[_Track]:
        """Tracks matched recently or still inside the recovery window."""
        return list(self._tracks)

    @property
    def trajectory_length(self) -> int:
        """Maximum number of trajectory points retained per track."""
        return max(1, self.config.track_buffer)

    def reset(self) -> None:
        """Forget every track and restart ID numbering (e.g. between videos)."""
        self._tracks = []
        self.frame_index = 0
        if self._tracker is not None:
            self._tracker.reset()
        logger.debug("Tracker state reset")

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """
        Update the tracker with this frame's detections.

        Args:
            detections: Detections found in the current frame.
            frame: Current video frame (used by the backend for motion cues).

        Returns:
            List[TrackedObject]: Tracks matched in this frame, with persistent IDs.
        """
        self.frame_index += 1
        boxes = self._to_boxes(detections, frame)
        raw = self._tracker.update(boxes, frame)
        matched = self._parse_output(raw, detections)
        self._sync(matched)
        return [_to_tracked_object(t) for t in self._tracks if t.track_id in matched]

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    def _to_boxes(self, detections: List[Detection], frame: Optional[np.ndarray]):
        """Build an ultralytics ``Boxes`` object (xyxy + conf + cls)."""
        if frame is not None:
            self._frame_shape = (int(frame.shape[0]), int(frame.shape[1]))

        if detections:
            data = np.array(
                [
                    [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
                    for d in detections
                ],
                dtype=np.float32,
            )
        else:
            data = np.zeros((0, 6), dtype=np.float32)
        return self._Boxes(data, self._frame_shape)

    @staticmethod
    def _parse_output(raw, detections: List[Detection]) -> Dict[int, _Observation]:
        """Turn the tracker's ``[xyxy, id, score, cls, idx]`` rows into observations."""
        if raw is None:
            return {}
        raw = np.asarray(raw)
        if raw.size == 0:
            return {}

        known_names = {d.class_id: d.class_name for d in detections}
        matched: Dict[int, _Observation] = {}
        for row in np.atleast_2d(raw):
            class_id = int(row[6])
            matched[int(row[4])] = _Observation(
                bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                confidence=float(row[5]),
                class_id=class_id,
                class_name=known_names.get(class_id, DEFAULT_CLASS_NAMES.get(class_id, "object")),
            )
        return matched

    def _sync(self, matched: Dict[int, _Observation]) -> None:
        """Merge this frame's observations into the internal track bookkeeping."""
        known_ids = {track.track_id for track in self._tracks}
        kept: List[_Track] = []

        for track in self._tracks:
            observation = matched.get(track.track_id)
            if observation is None:
                _age_track(track)
                # Discard tracks that outlived the recovery window or went stale.
                if track.time_since_update > self.config.track_buffer:
                    continue
                if track.score < self.config.track_low_thresh:
                    continue
                kept.append(track)
            else:
                _absorb(
                    track,
                    observation.bbox,
                    observation.confidence,
                    observation.class_id,
                    observation.class_name,
                    self.trajectory_length,
                )
                kept.append(track)

        # IDs handed out by BYTETracker are brand new to us on first sight.
        for track_id, observation in matched.items():
            if track_id not in known_ids:
                kept.append(_make_track(track_id, observation))

        kept.sort(key=lambda t: t.track_id)
        self._tracks = kept


def _load_bytetrack_args() -> Dict[str, object]:
    """
    Read Ultralytics' packaged ``bytetrack.yaml`` defaults.

    Returns an empty dict (callers then fall back to their own defaults) if the
    file cannot be found, so a layout change upstream never crashes a run.
    """
    try:
        import ultralytics
        import yaml

        cfg_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
        with open(cfg_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.warning("Could not read ultralytics bytetrack.yaml (%s) - using built-in defaults", exc)
    return {}


class _ByteTrackLite:
    """
    Dependency-free ByteTrack variant (the original implementation of this repo).

    Same two-stage association as the upstream tracker, but with a smoothed
    constant-velocity prediction instead of a Kalman filter and a pure-NumPy
    Hungarian solver instead of ``lap``.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: List[_Track] = []
        self.frame_index = 0
        # Monotonic: IDs are never handed out twice inside one run.
        self._next_id = 1
        logger.debug("ByteTrack-lite backend ready")

    @property
    def live_tracks(self) -> List[_Track]:
        """All tracks that have not been discarded yet (matched or temporarily lost)."""
        return list(self._tracks)

    @property
    def trajectory_length(self) -> int:
        """Maximum number of trajectory points retained per track."""
        return max(1, self.config.track_buffer)

    def reset(self) -> None:
        """Forget every track and restart ID numbering (e.g. between videos)."""
        self._tracks = []
        self.frame_index = 0
        self._next_id = 1
        logger.debug("Tracker state reset")

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """
        Update the tracker with this frame's detections.

        Args:
            detections: List of detections found in the current frame.
            frame: Optional current video frame (reserved for optical flow / GMC).

        Returns:
            List[TrackedObject]: Tracks matched in this frame, with persistent IDs.
        """
        self.frame_index += 1

        high = [d for d in detections if d.confidence >= self.config.track_high_thresh]
        low = [
            d
            for d in detections
            if self.config.track_low_thresh < d.confidence < self.config.track_high_thresh
        ]

        # Drop tracks that outlived the recovery window.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.config.track_buffer]

        predicted = [self._predict(t) for t in self._tracks]
        # Snapshot taken before stage 1 mutates anything: which tracks were live
        # and freshly matched when this frame started (index-aligned with _tracks).
        active_before = [t.time_since_update == 0 for t in self._tracks]

        # Stage 1: high-confidence detections against every live track.
        matches, u_tracks, u_dets = self._associate(
            self._tracks, predicted, high, self.config.match_thresh, fuse_score=True
        )
        for track_idx, det_idx in matches:
            detection = high[det_idx]
            _absorb(
                self._tracks[track_idx],
                detection.bbox,
                detection.confidence,
                detection.class_id,
                detection.class_name,
                self.trajectory_length,
            )

        # Stage 2: low-confidence detections against tracks still active this frame.
        # Lost tracks are deliberately excluded (ByteTrack sec. 3.2).
        candidates = [i for i in u_tracks if active_before[i]]
        matches2, _, _ = self._associate(
            [self._tracks[i] for i in candidates],
            [predicted[i] for i in candidates],
            low,
            STAGE2_MATCH_THRESH,
            fuse_score=False,
        )
        reactivated = {candidates[k] for k, _ in matches2}
        for track_idx, det_idx in matches2:
            detection = low[det_idx]
            _absorb(
                self._tracks[candidates[track_idx]],
                detection.bbox,
                detection.confidence,
                detection.class_id,
                detection.class_name,
                self.trajectory_length,
            )

        # Every track stage 1 missed and stage 2 could not rescue ages by one frame.
        for i in u_tracks:
            if i not in reactivated:
                _age_track(self._tracks[i])

        # Start new tracks from confident detections nobody claimed.
        for i in u_dets:
            detection = high[i]
            if detection.confidence >= self.config.new_track_thresh:
                track_id = self._next_id
                self._next_id += 1
                self._tracks.append(
                    _make_track(
                        track_id,
                        _Observation(
                            bbox=detection.bbox,
                            confidence=detection.confidence,
                            class_id=detection.class_id,
                            class_name=detection.class_name,
                        ),
                    )
                )
            else:
                logger.debug(
                    "Ignoring detection with confidence %.2f (< new_track_thresh %.2f)",
                    detection.confidence,
                    self.config.new_track_thresh,
                )

        # Permanently drop hopeless tracks.
        self._tracks = [
            t
            for t in self._tracks
            if t.time_since_update <= self.config.track_buffer
            and not (t.time_since_update > 0 and t.score < self.config.track_low_thresh)
        ]

        return [_to_tracked_object(t) for t in self._tracks if t.time_since_update == 0]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _predict(self, track: _Track) -> tuple[float, float, float, float]:
        """Shift the last known box by the smoothed velocity estimate."""
        x1, y1, x2, y2 = track.bbox
        vx, vy = track.velocity
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def _associate(
        self,
        tracks: List[_Track],
        predicted_boxes: List[tuple[float, float, float, float]],
        detections: List[Detection],
        thresh: float,
        fuse_score: bool = True,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Optimal one-to-one matching of tracks to detections.

        Args:
            tracks: Tracks to match.
            predicted_boxes: Motion-compensated boxes, index-aligned with ``tracks``.
            detections: Candidate detections for this stage.
            thresh: Maximum acceptable assignment cost.
            fuse_score: Blend the detection confidence into the cost (stage 1).

        Returns:
            (matches, unmatched_track_indices, unmatched_detection_indices)
        """
        if not tracks or not detections:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        ious = _iou_matrix(predicted_boxes, [d.bbox for d in detections])
        det_scores = np.array([d.confidence for d in detections], dtype=np.float64)

        # ByteTrack score fusion: cost = 1 - IoU * confidence. Stage 2 skips it -
        # with low scores the cost would always exceed its 0.5 limit.
        cost = 1.0 - ious * det_scores if fuse_score else 1.0 - ious

        track_classes = np.array([t.class_id for t in tracks], dtype=np.int64)
        det_classes = np.array([d.class_id for d in detections], dtype=np.int64)
        cross_class = track_classes[:, None] != det_classes[None, :]
        cost = np.where(cross_class, CLASS_MISMATCH_COST, cost)

        matches, u_tracks, u_dets = linear_assignment(cost, thresh)
        pairs = [(int(r), int(c)) for r, c in matches]
        return pairs, [int(i) for i in u_tracks], [int(i) for i in u_dets]


def _create_backend(config: TrackerConfig):
    """Instantiate the backend named by ``config.tracker_type``."""
    kind = str(config.tracker_type or BYTETRACK_BACKEND).strip().lower().replace("-", "_")

    if kind in {BYTETRACK_BACKEND, "ultralytics", "bytetrack_ultralytics"}:
        return _UltralyticsByteTrack(config)
    if kind in {BYTETRACK_LITE_BACKEND, "lite", "bytetrack_numpy"}:
        return _ByteTrackLite(config)

    raise ValueError(
        f"Unknown tracker_type '{config.tracker_type}'. "
        f"Valid options: '{BYTETRACK_BACKEND}' (Ultralytics built-in, needs lap) or "
        f"'{BYTETRACK_LITE_BACKEND}' (dependency-free fallback)."
    )


class PlayerTracker:
    """
    Maintains consistent player IDs across consecutive frames.

    This is a thin facade over the backend chosen by ``tracker_type``:

    * ``bytetrack`` - Ultralytics' ``BYTETracker`` (default), or
    * ``bytetrack_lite`` - the dependency-free NumPy implementation.

    ``update()`` returns only the tracks that were matched in the current frame,
    so objects briefly lost are simply not drawn until they reappear - they are
    not re-labelled in the meantime.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """
        Initialize the tracker.

        Args:
            config: Tracker configuration settings.

        Raises:
            ValueError: If ``config.tracker_type`` names an unknown backend.
            RuntimeError: If the selected backend cannot be imported.
        """
        self.config = config or TrackerConfig()
        self._impl = _create_backend(self.config)
        logger.info(
            "Initialized PlayerTracker (backend=%s, buffer=%d)",
            self.config.tracker_type,
            self.config.track_buffer,
        )

    @property
    def live_tracks(self) -> List[_Track]:
        """All tracks that have not been discarded yet (matched or temporarily lost)."""
        return self._impl.live_tracks

    @property
    def trajectory_length(self) -> int:
        """Maximum number of trajectory points retained per track."""
        return self._impl.trajectory_length

    @property
    def frame_index(self) -> int:
        """Number of frames pushed through the tracker since the last reset."""
        return self._impl.frame_index

    @property
    def backend(self) -> str:
        """Name of the configured backend."""
        return self.config.tracker_type

    def reset(self) -> None:
        """Forget every track and restart ID numbering (e.g. between videos)."""
        self._impl.reset()

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        Update tracker with new detections from the current frame.

        Args:
            detections: List of detections found in the current frame.
            frame: Optional current video frame (used for motion cues).

        Returns:
            List[TrackedObject]: Tracks matched in this frame, with persistent IDs.
        """
        return self._impl.update(detections, frame)
