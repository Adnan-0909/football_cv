"""
Interactive Pitch Calibration (Stage 4)
=======================================

Lets you pick **corresponding points** between a video frame and a standardized
top-down pitch diagram, then stores them so :class:`app.pitch.PitchTransformer`
can estimate the homography with ``cv2.findHomography()``.

How it works (``python main.py --calibrate``)
---------------------------------------------

1. A frame of the configured input video is shown (default: the middle frame,
   override with ``--calibration-frame N``).
2. **Phase 1 - image**: left-click landmarks that are *on the ground* of the
   real pitch - penalty-box corners, the centre spot, the halfway-line/sideline
   intersections, ground-level ad-board corners.  Click them in any order, but
   note the numbers; you will repeat them in the same order on the pitch
   diagram.  Aim for **4-8 points spread across the whole visible pitch**
   (never players, sky, stands or camera shakes - only things whose real-world
   position you know).  Press **Enter** to continue.
3. **Phase 2 - pitch**: click the *same* landmarks on the drawn pitch diagram
   in the same order.  Press **Enter** to build the homography.
4. The result is saved to ``pitch.calibration_path`` (default
   ``calibration/pitch.json``) together with the mean reprojection error - a
   small value (well under ~1 m) means a good calibration.

Controls
--------
    left click   add point (current phase)
    u            undo last point
    r            restart from phase 1
    Enter        finish current phase / build + save
    ESC / q      cancel

Why manual?  Automatic pitch-line detection is intentionally *not* part of
Stage 4; :mod:`app.pitch` keeps the pipeline decoupled from how the homography
is obtained, so a smarter calibrator can be swapped in later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from app.config import PitchConfig
from app.pitch import PitchTransformer, draw_pitch

logger = logging.getLogger(__name__)

# Rendered pitch diagram size (pixels). 10 px per meter on a 105x68 pitch.
DIAGRAM_SIZE: Tuple[int, int] = (1050, 680)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class CalibrationSession:
    """
    GUI-free state machine behind the calibration windows.

    Collects image points (phase 1) and the matching pitch points (phase 2),
    then builds a calibrated :class:`~app.pitch.PitchTransformer`.

    The pitch phase converts diagram pixels into meters with the documented
    coordinate system: ``x = px / diagram_width * length``,
    ``y = py / diagram_height * width``.
    """

    def __init__(
        self,
        config: PitchConfig,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        frame_index: Optional[int] = None,
        diagram_size: Tuple[int, int] = DIAGRAM_SIZE,
    ) -> None:
        self.config = config
        self.image_size = image_size
        self.frame_index = frame_index
        self.diagram_size = diagram_size
        self.image_points: List[Tuple[float, float]] = []
        self.pitch_points: List[Tuple[float, float]] = []
        self.phase = "image"  # "image" -> "pitch" -> "done"
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #

    @property
    def min_points(self) -> int:
        return int(self.config.min_points)

    @property
    def progress_text(self) -> str:
        """Short status string shown in the window title / overlay."""
        if self.phase == "image":
            return f"image points {len(self.image_points)} (need >= {self.min_points})"
        if self.phase == "pitch":
            return (
                f"pitch points {len(self.pitch_points)}/{len(self.image_points)} "
                "- click in the same order"
            )
        return "done"

    def add_point(self, x: float, y: float) -> None:
        """Add a click for the current phase (pixels in that window)."""
        if self.phase == "done":
            return
        if self.phase == "image":
            self.image_points.append((float(x), float(y)))
            self.error = None
            return
        # Phase 2: diagram pixels -> pitch meters (see module docstring).
        width_px, height_px = self.diagram_size
        pitch_x = float(x) / width_px * float(self.config.length_meters)
        pitch_y = float(y) / height_px * float(self.config.width_meters)
        self.pitch_points.append((pitch_x, pitch_y))
        self.error = None

    def undo(self) -> None:
        """Remove the most recent point of the current phase."""
        if self.phase == "image" and self.image_points:
            self.image_points.pop()
        elif self.phase == "pitch" and self.pitch_points:
            self.pitch_points.pop()

    def reset(self) -> None:
        """Back to phase 1 with no points."""
        self.image_points.clear()
        self.pitch_points.clear()
        self.phase = "image"
        self.error = None

    def advance(self) -> bool:
        """
        Finish the current phase.

        Returns:
            True when the session is complete (homography estimated).

        Raises:
            ValueError: Too few points so far - ``self.error`` explains why.
        """
        if self.phase == "image":
            if len(self.image_points) < self.min_points:
                self.error = (
                    f"Need at least {self.min_points} points, "
                    f"clicked {len(self.image_points)}"
                )
                raise ValueError(self.error)
            self.phase = "pitch"
            self.error = None
            return False

        if self.phase == "pitch":
            if len(self.pitch_points) != len(self.image_points):
                self.error = (
                    f"Click one pitch point per image point "
                    f"({len(self.image_points)} expected, {len(self.pitch_points)} done)"
                )
                raise ValueError(self.error)
            self.phase = "done"
            self.error = None
            return True

        return True

    def build(self) -> PitchTransformer:
        """Estimate and return a calibrated transformer from the picked pairs."""
        transformer = PitchTransformer(self.config)
        transformer.estimate_homography(self.image_points, self.pitch_points)
        transformer.image_size = self.image_size
        transformer.frame_index = self.frame_index
        return transformer


# ---------------------------------------------------------------------- #
# Rendering helpers (pure functions - unit-testable without a window)
# ---------------------------------------------------------------------- #

def draw_image_phase(frame: np.ndarray, session: CalibrationSession) -> np.ndarray:
    """Copy of the frame with numbered markers and on-screen instructions."""
    canvas = frame.copy()
    _draw_banner(
        canvas,
        [
            "PHASE 1/2 - click points ON THE GROUND (penalty corners, centre spot, ...)",
            f"  {session.progress_text}   [Enter] next   [u] undo   [r] reset   [Esc] cancel",
        ],
    )
    for i, (x, y) in enumerate(session.image_points, start=1):
        _draw_marker(canvas, int(round(x)), int(round(y)), i)
    if session.error:
        _draw_banner(canvas, [session.error], top=False, color=(0, 0, 255))
    return canvas


def draw_pitch_phase(session: CalibrationSession) -> np.ndarray:
    """Top-down pitch diagram with the pitch-phase picks (and pair lines)."""
    diagram = draw_pitch(
        session.config.length_meters, session.config.width_meters, session.diagram_size
    )
    width_px, height_px = session.diagram_size
    points_px: List[Tuple[int, int]] = []
    for pitch_x, pitch_y in session.pitch_points:
        px = int(round(pitch_x / session.config.length_meters * width_px))
        py = int(round(pitch_y / session.config.width_meters * height_px))
        points_px.append((px, py))

    # Pair each pick with its image-phase counterpart (same index/order).
    for i, px in enumerate(points_px, start=1):
        if i <= len(session.image_points):
            _draw_marker(diagram, px[0], px[1], i, color=(0, 255, 255))
    for a, b in zip(points_px, points_px[1:]):
        cv2.line(diagram, a, b, (0, 255, 255), 1, cv2.LINE_AA)

    # Ghost markers showing where the remaining image points must be picked.
    for i in range(len(points_px), len(session.image_points)):
        _draw_marker(diagram, width_px - 30, height_px - 30 - 22 * i, i + 1, color=(200, 200, 200))

    _draw_banner(
        diagram,
        [
            "PHASE 2/2 - click the SAME points on the pitch, same order",
            f"  {session.progress_text}   [Enter] build + save   [u] undo   [r] restart   [Esc] cancel",
        ],
    )
    if session.error:
        _draw_banner(diagram, [session.error], top=False, color=(0, 0, 255))
    return diagram


def _draw_marker(canvas: np.ndarray, x: int, y: int, index: int, color=(0, 255, 0)) -> None:
    """Numbered crosshair marker."""
    cv2.circle(canvas, (x, y), 9, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x - 14, y), (x + 14, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 14), (x, y + 14), color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas, str(index), (x + 12, y - 10), _FONT, 0.6, color, 2, cv2.LINE_AA
    )


def _draw_banner(
    canvas: np.ndarray, lines: Sequence[str], *, top: bool = True, color=(0, 255, 255)
) -> None:
    """Instruction banner pinned to the top (or bottom) of the window."""
    pad, line_h = 6, 22
    box_h = pad * 2 + line_h * len(lines)
    overlay = canvas.copy()
    y0 = 0 if top else canvas.shape[0] - box_h
    cv2.rectangle(overlay, (0, y0), (canvas.shape[1], y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    for i, text in enumerate(lines):
        cv2.putText(
            canvas, text, (10, y0 + pad + line_h * (i + 1) - 6),
            _FONT, 0.55, color, 1, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------- #
# Interactive entry point
# ---------------------------------------------------------------------- #

def pick_frame(video_path: Union[str, Path], frame_index: Optional[int] = None) -> np.ndarray:
    """
    Read one frame from the video (default: the middle frame).

    Raises:
        FileNotFoundError: Video missing.
        IOError: Video cannot be opened / shorter than expected.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = int(frame_index) if frame_index is not None else max(0, total // 2)
        index = max(0, min(index, max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IOError(f"Could not read frame {index} from {path}")
        frame = np.ascontiguousarray(frame)
        logger.info("Calibrating against frame %d of %s (%d frames total)", index, path, total)
        return frame
    finally:
        cap.release()


def run_calibration(
    config: PitchConfig,
    video_path: Union[str, Path],
    frame_index: Optional[int] = None,
) -> Optional[Path]:
    """
    Open the two calibration windows, collect points, save the calibration.

    Phase 1 clicks land on the video frame, phase 2 on the pitch diagram;
    **Enter** advances, **Esc** cancels (returns None).

    Returns:
        Path of the written calibration file, or None when cancelled.

    Raises:
        RuntimeError: If no GUI is available (headless session).
    """
    frame = pick_frame(video_path, frame_index)
    session = CalibrationSession(
        config, image_size=(frame.shape[1], frame.shape[0]), frame_index=frame_index
    )

    image_win = "Pitch calibration - phase 1 (video frame)"
    pitch_win = "Pitch calibration - phase 2 (pitch diagram)"

    def refresh() -> None:
        if session.phase == "image":
            cv2.imshow(image_win, draw_image_phase(frame, session))
            try:
                cv2.destroyWindow(pitch_win)
            except cv2.error:
                pass
        else:
            cv2.imshow(pitch_win, draw_pitch_phase(session))
            try:
                cv2.destroyWindow(image_win)
            except cv2.error:
                pass

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            session.add_point(x, y)
            refresh()

    try:
        cv2.namedWindow(image_win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(image_win, on_mouse)
        cv2.namedWindow(pitch_win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(pitch_win, on_mouse)
    except cv2.error as exc:  # pragma: no cover - headless environments
        raise RuntimeError(
            "Could not open calibration windows (no GUI available). "
            "Run this on a desktop session."
        ) from exc

    refresh()
    cancelled = False
    saved: Optional[Path] = None
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):  # ESC / q
            cancelled = True
            break
        elif key in (ord("u"), ord("U")):
            session.undo()
            refresh()
        elif key in (ord("r"), ord("R")):
            session.reset()
            refresh()
        elif key in (13, 10):  # Enter
            try:
                done = session.advance()
            except ValueError:
                refresh()
                continue
            if done:
                transformer = session.build()
                saved = transformer.save_calibration(
                    config.calibration_path,
                    frame_index=session.frame_index,
                    image_size=session.image_size,
                )
                logger.info(
                    "Calibration saved: %d points, mean reprojection error %.3f m -> %s",
                    transformer.point_count,
                    transformer.mean_error_meters or 0.0,
                    saved,
                )
                _show_reprojection(frame, session, transformer, image_win)
                break
            refresh()

    cv2.destroyAllWindows()
    if cancelled:
        logger.info("Calibration cancelled - nothing saved.")
        return None
    return saved


def _show_reprojection(
    frame: np.ndarray,
    session: CalibrationSession,
    transformer: PitchTransformer,
    window_name: str,
) -> None:  # pragma: no cover - visual feedback only
    """Overlay picked vs reprojected points so a bad fit is visible at once."""
    canvas = frame.copy()
    image_pts = np.asarray(session.image_points, dtype=np.float64)
    projected = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2), transformer.homography_matrix
    ).reshape(-1, 2)
    for (x, y), (px, py), i in zip(image_pts, projected, range(1, len(image_pts) + 1)):
        cv2.line(canvas, (int(x), int(y)), (int(px), int(py)), (0, 0, 255), 1, cv2.LINE_AA)
        _draw_marker(canvas, int(px), int(py), i, color=(255, 255, 0))
    _draw_banner(
        canvas,
        [
            "REPROJECTION CHECK - cyan = your clicks, yellow = model estimate",
            f"  mean error {transformer.mean_error_meters:.2f} m   [any key] close",
        ],
    )
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
