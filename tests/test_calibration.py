"""
Unit Tests for Interactive Pitch Calibration (Stage 4)
=====================================================
Covers the GUI-free CalibrationSession state machine, the window rendering
helpers, and frame picking - no GUI is opened here.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.calibration import (
    DIAGRAM_SIZE,
    CalibrationSession,
    draw_image_phase,
    draw_pitch_phase,
    pick_frame,
)
from app.config import PitchConfig
from app.pitch import PitchTransformer

IMAGE_POINTS = [(50.0, 60.0), (270.0, 55.0), (300.0, 190.0), (30.0, 200.0)]
# Same landmarks as the image corners would be on the pitch (meters).
PITCH_POINTS = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]


def make_session(config: PitchConfig = None) -> CalibrationSession:
    return CalibrationSession(
        config or PitchConfig(),
        image_size=(320, 240),
        frame_index=0,
    )


def fill_both_phases(session: CalibrationSession) -> None:
    for point in IMAGE_POINTS:
        session.add_point(*point)
    session.advance()
    width_px, height_px = session.diagram_size
    for pitch_x, pitch_y in PITCH_POINTS:
        session.add_point(
            pitch_x / session.config.length_meters * width_px,
            pitch_y / session.config.width_meters * height_px,
        )


# ---------------------------------------------------------------------- #
# Session state machine
# ---------------------------------------------------------------------- #

def test_session_starts_in_image_phase():
    session = make_session()
    assert session.phase == "image"
    assert session.image_points == [] and session.pitch_points == []


def test_advance_requires_minimum_image_points():
    session = make_session()
    for point in IMAGE_POINTS[:2]:
        session.add_point(*point)
    with pytest.raises(ValueError, match="at least 4"):
        session.advance()
    assert session.phase == "image" and session.error


def test_advance_image_to_pitch_phase():
    session = make_session()
    for point in IMAGE_POINTS:
        session.add_point(*point)
    assert session.advance() is False
    assert session.phase == "pitch"


def test_pitch_clicks_convert_diagram_pixels_to_meters():
    """The diagram origin is pitch (0,0); its centre is (length/2, width/2)."""
    session = make_session()
    for point in IMAGE_POINTS:
        session.add_point(*point)
    session.advance()

    width_px, height_px = DIAGRAM_SIZE
    session.add_point(width_px / 2, height_px / 2)
    assert session.pitch_points[0] == pytest.approx((52.5, 34.0))

    session.add_point(0, 0)
    assert session.pitch_points[1] == pytest.approx((0.0, 0.0))


def test_advance_pitch_phase_requires_matching_counts():
    session = make_session()
    for point in IMAGE_POINTS[:3]:
        session.add_point(*point)
    # Bypass the minimum check by adding a 4th, then mismatch on phase 2.
    session.add_point(IMAGE_POINTS[3][0], IMAGE_POINTS[3][1])
    session.advance()
    session.add_point(10.0, 10.0)  # only 1 of 4 pitch points
    with pytest.raises(ValueError, match="one pitch point per image point"):
        session.advance()
    assert session.phase == "pitch"


def test_undo_and_reset():
    session = make_session()
    for point in IMAGE_POINTS:
        session.add_point(*point)
    session.undo()
    assert len(session.image_points) == 3
    session.add_point(*IMAGE_POINTS[-1])  # restore to the minimum
    session.advance()

    session.add_point(100.0, 100.0)
    session.undo()
    assert session.pitch_points == []

    session.reset()
    assert session.phase == "image"
    assert session.image_points == []


def test_build_returns_calibrated_transformer(tmp_path: Path):
    session = make_session()
    fill_both_phases(session)
    assert session.advance() is True
    assert session.phase == "done"

    transformer = session.build()
    assert isinstance(transformer, PitchTransformer)
    assert transformer.is_ready
    assert transformer.mean_error_meters < 0.01  # perfect clicks -> no error
    assert transformer.image_size == (320, 240)
    assert transformer.frame_index == 0

    # The built transformer projects the picked image points back correctly.
    for (u, v), (px, py) in zip(IMAGE_POINTS, PITCH_POINTS):
        assert transformer.image_to_pitch(u, v) == pytest.approx((px, py), abs=1e-6)


# ---------------------------------------------------------------------- #
# Rendering helpers (no GUI)
# ---------------------------------------------------------------------- #

def test_draw_image_phase_does_not_mutate_frame():
    session = make_session()
    for point in IMAGE_POINTS[:2]:
        session.add_point(*point)
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)

    rendered = draw_image_phase(frame, session)

    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, frame), "markers/instructions should be drawn"
    assert np.array_equal(frame, np.full((240, 320, 3), 40, np.uint8)), "input untouched"


def test_draw_pitch_phase_shape_and_progress():
    session = make_session()
    for point in IMAGE_POINTS:
        session.add_point(*point)
    session.advance()

    rendered = draw_pitch_phase(session)
    assert rendered.shape == (DIAGRAM_SIZE[1], DIAGRAM_SIZE[0], 3)
    assert "pitch points 0/4" in session.progress_text


# ---------------------------------------------------------------------- #
# Frame picking (no GUI)
# ---------------------------------------------------------------------- #

def test_pick_frame_reads_middle_frame(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120)
    )
    assert writer.isOpened()
    for i in range(5):
        frame = np.full((120, 160, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frame = pick_frame(path)
    assert frame.shape == (120, 160, 3)

    first = pick_frame(path, frame_index=0)
    assert first.shape == (120, 160, 3)


def test_pick_frame_missing_video(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pick_frame(tmp_path / "missing.mp4")
