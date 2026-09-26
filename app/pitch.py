"""
2D Pitch Mapping Module (Stage 4)
=================================

Transforms perspective camera coordinates into **top-down football pitch
coordinates (meters)** using a manually calibrated homography.

Coordinate system
-----------------
The pitch is a rectangle defined by :class:`~app.config.PitchConfig`::

    (0, 0) --------------------->  x = length_meters (105 m)
      |                                (right goal line)
      |
      v
    y = width_meters (68 m)
      (far sideline in the top-down rendering)

* **Origin (0, 0)** is the bottom-left corner of the pitch as drawn in the
  top-down view (i.e. the left end of the *near* sideline).
* **x** runs along the pitch *length*: 0 = left goal line, 105 = right goal
  line.  ``length_meters`` is configurable (config.yaml -> ``pitch``).
* **y** runs along the pitch *width*: 0 = near sideline (top of the rendering
  grows downward, OpenCV image convention), 68 = far sideline.
* **Units are always meters.**

Foot position
-------------
Players are projected using their *foot* contact point, not the bbox centre -
the centre floats in mid-air and would distort the mapping::

    foot_x = (x1 + x2) / 2      # horizontal centre of the bounding box
    foot_y = y2                 # bottom edge of the bounding box

See :func:`foot_position`.

Manual calibration (no automatic pitch detection)
-------------------------------------------------
Corresponding points are picked between a video frame and a standardized
pitch diagram (``python main.py --calibrate``) and stored in a JSON file
(``pitch.calibration_path``, default ``calibration/pitch.json``)::

    image (pixels)  <->  pitch (meters)
    P1 = (u1, v1)       P1 = (x1, y1)
    ...                  ...
    P4 = (u4, v4)       P4 = (x4, y4)

:func:`PitchTransformer.estimate_homography` feeds those pairs to
``cv2.findHomography()``; :meth:`PitchTransformer.image_to_pitch` applies the
resulting perspective transform.  Pick points that lie **on the pitch plane**
(pitch markings, ground-level ad boards) - never players, sky or stands.

Modularity
----------
Future automatic camera calibration can replace the manual step by writing
the same calibration file (or calling ``calibrate()``); nothing downstream
changes::

    transformer = PitchTransformer(config.pitch)
    transformer.load_calibration("calibration/pitch.json")
    pitch_x, pitch_y = transformer.image_to_pitch(foot_x, foot_y)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from app.config import PitchConfig

logger = logging.getLogger(__name__)

# JSON schema version of the calibration file.
CALIBRATION_VERSION = 1

# Pitch markings (meters, FIFA standard) used by draw_pitch().
_PENALTY_BOX_DEPTH = 16.5
_PENALTY_BOX_WIDTH = 40.32
_GOAL_AREA_DEPTH = 5.5
_GOAL_AREA_WIDTH = 18.32
_PENALTY_SPOT = 11.0
_CENTER_RADIUS = 9.15
_CORNER_RADIUS = 1.0


@dataclass
class PitchCoordinate:
    """
    Position on a standard 2D football pitch in meters.

    Attributes:
        x: Distance along length (0.0 to 105.0 meters).
        y: Distance along width (0.0 to 68.0 meters).
    """
    x: float
    y: float


def foot_position(bbox: Sequence[float]) -> Tuple[float, float]:
    """
    Foot contact point of a bounding box (what actually touches the grass).

    Args:
        bbox: (x1, y1, x2, y2) bounding box in image pixels.

    Returns:
        (x, y): horizontal centre of the box and its *bottom* edge, i.e.
        ``x = (x1 + x2) / 2``, ``y = y2``.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, y2


class PitchTransformer:
    """
    Projects pixel coordinates from broadcast camera view onto 2D football
    pitch coordinates using a manual homography calibration.
    """

    def __init__(self, config: Optional[PitchConfig] = None) -> None:
        """
        Args:
            config: Pitch dimensions configuration (length, width in meters).
        """
        self.config = config or PitchConfig()
        self.homography_matrix: Optional[np.ndarray] = None
        # Raw calibration pairs (kept for re-saving / inspection).
        self.image_points: Optional[np.ndarray] = None
        self.pitch_points: Optional[np.ndarray] = None
        # Metadata from the calibration file (informational only).
        self.frame_index: Optional[int] = None
        self.image_size: Optional[Tuple[int, int]] = None
        self.mean_error_meters: Optional[float] = None
        logger.info(
            "Initialized PitchTransformer (pitch_size=%.1fm x %.1fm, calibration=%s)",
            self.config.length_meters,
            self.config.width_meters,
            self.config.calibration_path,
        )

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        """True once a homography has been estimated / loaded."""
        return self.homography_matrix is not None

    @property
    def point_count(self) -> int:
        """Number of image/pitch correspondences used by the homography."""
        if self.image_points is None:
            return 0
        return int(self.image_points.shape[0])

    # ------------------------------------------------------------------ #
    # Homography
    # ------------------------------------------------------------------ #

    def estimate_homography(
        self,
        image_points: Sequence[Sequence[float]],
        pitch_points: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Compute the 3x3 homography from corresponding points.

        Args:
            image_points: Nx2 pixel coordinates (image space).
            pitch_points: Nx2 coordinates on the pitch, in meters.

        Returns:
            np.ndarray: 3x3 homography matrix (image -> pitch meters).

        Raises:
            ValueError: Fewer than ``config.min_points`` pairs, mismatched
                lengths, non-finite values, or (nearly) collinear points.
            RuntimeError: If OpenCV fails to estimate a homography.
        """
        img = np.asarray(image_points, dtype=np.float64)
        pit = np.asarray(pitch_points, dtype=np.float64)

        if img.ndim != 2 or img.shape[1] != 2 or pit.shape != img.shape:
            raise ValueError(
                f"Expected matching Nx2 point arrays, got image={img.shape}, pitch={pit.shape}"
            )
        if img.shape[0] < self.config.min_points:
            raise ValueError(
                f"Need at least {self.config.min_points} image/pitch point pairs, "
                f"got {img.shape[0]}. Pick well-spread points on the pitch plane."
            )
        if not (np.isfinite(img).all() and np.isfinite(pit).all()):
            raise ValueError("Calibration points must be finite numbers")

        self._reject_collinear(img, "image points")
        self._reject_collinear(pit, "pitch points")

        method = cv2.RANSAC if img.shape[0] > 4 else 0
        homography, _mask = cv2.findHomography(img, pit, method)
        if homography is None or not np.isfinite(homography).all():
            raise RuntimeError(
                "cv2.findHomography() failed - check that the paired points are "
                "correct and not nearly collinear."
            )

        self.homography_matrix = homography
        self.image_points = img
        self.pitch_points = pit
        self.mean_error_meters = self._reprojection_error(img, pit, homography)
        logger.info(
            "Homography estimated from %d points (mean reprojection error: %.3f m)",
            img.shape[0],
            self.mean_error_meters,
        )
        return homography

    @staticmethod
    def _reject_collinear(points: np.ndarray, name: str) -> None:
        """Reject point sets that lie on a single line (homography undefined)."""
        centered = points - points.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular.size < 2 or singular[1] <= max(1e-6, 1e-6 * singular[0]):
            raise ValueError(
                f"{name} are collinear - spread the calibration points over the "
                "pitch (e.g. both penalty boxes and the halfway line)."
            )

    @staticmethod
    def _reprojection_error(
        image_points: np.ndarray, pitch_points: np.ndarray, homography: np.ndarray
    ) -> float:
        """Mean distance (meters) between projected and picked pitch points."""
        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        return float(np.linalg.norm(projected - pitch_points, axis=1).mean())

    # ------------------------------------------------------------------ #
    # Transformation
    # ------------------------------------------------------------------ #

    def image_to_pitch(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform one image coordinate into pitch coordinates (meters).

        Args:
            x: Horizontal pixel coordinate (typically the foot position).
            y: Vertical pixel coordinate (bottom of the bounding box).

        Returns:
            (pitch_x, pitch_y): position on the pitch in meters.

        Raises:
            RuntimeError: If no calibration/homography is loaded yet.
            ValueError: If ``x``/``y`` are not finite numbers.
        """
        if self.homography_matrix is None:
            raise RuntimeError(
                "PitchTransformer is not calibrated - run "
                "'python main.py --calibrate' and pick the correspondence points "
                "first (or load a calibration file)."
            )
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Non-finite image coordinate: ({x}, {y})")

        projected = cv2.perspectiveTransform(
            np.array([[[float(x), float(y)]]], dtype=np.float64),
            self.homography_matrix,
        )[0, 0]
        return float(projected[0]), float(projected[1])

    def transform_point(self, pixel_x: float, pixel_y: float) -> PitchCoordinate:
        """
        Transform a single image coordinate (foot position) to the pitch.

        Args:
            pixel_x: Horizontal pixel coordinate.
            pixel_y: Vertical pixel coordinate (feet contact point).

        Returns:
            PitchCoordinate: (x, y) on the 2D pitch in meters.
        """
        pitch_x, pitch_y = self.image_to_pitch(pixel_x, pixel_y)
        return PitchCoordinate(x=pitch_x, y=pitch_y)

    # ------------------------------------------------------------------ #
    # Calibration persistence
    # ------------------------------------------------------------------ #

    def save_calibration(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        frame_index: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Path:
        """
        Write the current calibration pairs to a JSON file.

        The homography itself is *not* stored - it is always recomputed from
        the points on load, so the file stays valid and human-readable.

        Raises:
            RuntimeError: If no calibration has been estimated yet.
        """
        if self.image_points is None or self.pitch_points is None:
            raise RuntimeError("No calibration to save - estimate the homography first.")
        destination = Path(path) if path is not None else Path(self.config.calibration_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": CALIBRATION_VERSION,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pitch": {
                "length_meters": float(self.config.length_meters),
                "width_meters": float(self.config.width_meters),
            },
            "image_size": list(self.image_size) if self.image_size else None,
            "frame_index": self.frame_index if frame_index is None else frame_index,
            "image_points": self.image_points.tolist(),
            "pitch_points": self.pitch_points.tolist(),
            "mean_error_meters": self.mean_error_meters,
        }
        if image_size is not None:
            payload["image_size"] = list(image_size)
        if frame_index is not None:
            payload["frame_index"] = int(frame_index)

        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Saved pitch calibration (%d points) to %s", self.point_count, destination
        )
        return destination

    def load_calibration(self, path: Optional[Union[str, Path]] = None) -> "PitchTransformer":
        """
        Load a calibration file and rebuild the homography from its points.

        Raises:
            FileNotFoundError: File does not exist.
            ValueError: File is malformed / too few points / collinear.
            RuntimeError: Pitch dimensions in the file differ from the config.
        """
        source = Path(path) if path is not None else Path(self.config.calibration_path)
        if not source.exists():
            raise FileNotFoundError(f"Calibration file not found: {source}")

        payload = json.loads(source.read_text(encoding="utf-8"))
        pitch = payload.get("pitch") or {}
        file_length = float(pitch.get("length_meters", self.config.length_meters))
        file_width = float(pitch.get("width_meters", self.config.width_meters))
        if not (
            math.isclose(file_length, float(self.config.length_meters))
            and math.isclose(file_width, float(self.config.width_meters))
        ):
            raise RuntimeError(
                f"Calibration {source} was made for a {file_length:g}x{file_width:g} m pitch "
                f"but config.yaml says {self.config.length_meters:g}x"
                f"{self.config.width_meters:g} m - align them (or recalibrate)."
            )

        self.estimate_homography(payload["image_points"], payload["pitch_points"])
        self.frame_index = payload.get("frame_index")
        size = payload.get("image_size")
        self.image_size = tuple(size) if size else None
        logger.info(
            "Loaded pitch calibration from %s (frame=%s, mean error=%.3f m)",
            source,
            self.frame_index,
            self.mean_error_meters or 0.0,
        )
        return self


def draw_pitch(
    length_meters: float,
    width_meters: float,
    size: Tuple[int, int] = (700, 450),
    *,
    background: Tuple[int, int, int] = (45, 120, 45),
    line_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: Optional[int] = None,
) -> np.ndarray:
    """
    Render a top-down pitch diagram with the standard markings.

    Mapping: ``px = x / length * width_px`` and ``py = y / height_px`` - the
    pitch coordinate origin (0, 0) is the top-left pixel of the rendering and
    +y grows downward, exactly like the coordinate system documented above.

    Args:
        length_meters: Pitch length in meters (x axis, 0 -> length).
        width_meters: Pitch width in meters (y axis, 0 -> width).
        size: (width, height) of the rendered image in pixels.
        background: Grass colour (BGR).
        line_color: Marking colour (BGR).
        thickness: Line thickness (auto-scaled when None).

    Returns:
        np.ndarray: size[1] x size[0] x 3 diagram.
    """
    width_px, height_px = int(size[0]), int(size[1])
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid pitch diagram size: {size}")
    canvas = np.full((height_px, width_px, 3), background, dtype=np.uint8)

    sx = width_px / float(length_meters)
    sy = height_px / float(width_meters)
    # Uniform scale for circles/arcs so the diagram stays undistorted.
    s = min(sx, sy)
    if thickness is None:
        thickness = max(1, int(round(min(width_px, height_px) / 220)))

    def pt(x_m: float, y_m: float) -> Tuple[int, int]:
        return int(round(x_m * sx)), int(round(y_m * sy))

    def line(a: Tuple[float, float], b: Tuple[float, float]) -> None:
        cv2.line(canvas, pt(*a), pt(*b), line_color, thickness)

    def rect(x1: float, y1: float, x2: float, y2: float) -> None:
        cv2.rectangle(canvas, pt(x1, y1), pt(x2, y2), line_color, thickness)

    def circle(center: Tuple[float, float], radius_m: float, filled: bool = False) -> None:
        radius_px = int(round(radius_m * s))
        if filled:
            radius_px = max(1, radius_px)  # spots must stay visible at any size
        cv2.circle(canvas, pt(*center), radius_px, line_color, -1 if filled else thickness)

    length, width = float(length_meters), float(width_meters)
    half_w, half_l = width / 2.0, length / 2.0

    # Outer boundary + halfway line.
    rect(0.0, 0.0, length, width)
    line((half_l, 0.0), (half_l, width))
    circle((half_l, half_w), _CENTER_RADIUS)
    circle((half_l, half_w), 0.12, filled=True)

    for side in (0.0, 1.0):  # left (x=0) and right (x=length) goal sides
        goal_x = 0.0 if side == 0.0 else length
        sign = 1.0 if side == 0.0 else -1.0
        # Penalty area: 16.5 m deep, 40.32 m wide, centred on the goal.
        rect(
            goal_x,
            half_w - _PENALTY_BOX_WIDTH / 2.0,
            goal_x + sign * _PENALTY_BOX_DEPTH,
            half_w + _PENALTY_BOX_WIDTH / 2.0,
        )
        # Goal area: 5.5 m deep, 18.32 m wide.
        rect(
            goal_x,
            half_w - _GOAL_AREA_WIDTH / 2.0,
            goal_x + sign * _GOAL_AREA_DEPTH,
            half_w + _GOAL_AREA_WIDTH / 2.0,
        )
        circle((goal_x + sign * _PENALTY_SPOT, half_w), 0.12, filled=True)

    # Corner arcs (1 m radius).
    r = _CORNER_RADIUS
    arcs = (
        ((0.0, 0.0), 0, 90),
        ((length, 0.0), 90, 180),
        ((length, width), 180, 270),
        ((0.0, width), 270, 360),
    )
    for corner, start, end in arcs:
        center_px = pt(*corner)
        # cv2.ellipse angles are clockwise from +x with y downward; offset the
        # quadrant so each arc bends *into* the pitch.
        cv2.ellipse(
            canvas,
            center_px,
            (int(round(r * s)), int(round(r * s))),
            0,
            start,
            end,
            line_color,
            thickness,
        )

    return canvas
