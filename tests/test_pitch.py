"""
Unit Tests for Stage 4 Pitch Mapping
====================================
Covers foot-position math, homography estimation (cv2.findHomography),
image_to_pitch() in meters, calibration save/load, and the top-down diagram.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import PitchConfig
from app.pitch import PitchCoordinate, PitchTransformer, draw_pitch, foot_position

# A simple 1:1-ish camera: frame corners <-> pitch corners (y grows downward
# in both spaces, so the mapping is well defined).
FRAME_W, FRAME_H = 320, 240
IMAGE_CORNERS = [(0.0, 0.0), (FRAME_W, 0.0), (FRAME_W, FRAME_H), (0.0, FRAME_H)]
PITCH_CORNERS = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]


def make_transformer(config: PitchConfig = None) -> PitchTransformer:
    transformer = PitchTransformer(config or PitchConfig())
    transformer.estimate_homography(IMAGE_CORNERS, PITCH_CORNERS)
    return transformer


# ---------------------------------------------------------------------- #
# Foot position
# ---------------------------------------------------------------------- #

def test_foot_position_is_bottom_center():
    """x = horizontal centre, y = bottom edge of the bounding box."""
    x, y = foot_position((10.0, 20.0, 50.0, 120.0))
    assert x == pytest.approx(30.0)
    assert y == pytest.approx(120.0)


def test_foot_position_not_bbox_center():
    """The foot point must differ from the (floating) bbox centre."""
    x1, y1, x2, y2 = 0.0, 0.0, 40.0, 100.0
    fx, fy = foot_position((x1, y1, x2, y2))
    assert fy != pytest.approx((y1 + y2) / 2.0)


# ---------------------------------------------------------------------- #
# Homography estimation
# ---------------------------------------------------------------------- #

def test_estimate_homography_maps_corners_exactly():
    """With exactly 4 points the mapping is exact at those points."""
    transformer = make_transformer()
    for (u, v), (px, py) in zip(IMAGE_CORNERS, PITCH_CORNERS):
        got_x, got_y = transformer.image_to_pitch(u, v)
        assert got_x == pytest.approx(px, abs=1e-6)
        assert got_y == pytest.approx(py, abs=1e-6)
    assert transformer.mean_error_meters == pytest.approx(0.0, abs=1e-6)


def test_estimate_homography_perspective_roundtrip():
    """Recover a known projective mapping (image generated from pitch)."""
    camera = np.array(
        [[8.0, 1.5, 100.0], [0.5, 6.0, 50.0], [0.001, 0.002, 1.0]], dtype=np.float64
    )
    pitch_pts = np.array(PITCH_CORNERS, dtype=np.float64)
    image_pts = cv2.perspectiveTransform(pitch_pts.reshape(-1, 1, 2), camera).reshape(-1, 2)

    transformer = PitchTransformer()
    transformer.estimate_homography(image_pts, pitch_pts)
    for (u, v), (px, py) in zip(image_pts, pitch_pts):
        got_x, got_y = transformer.image_to_pitch(float(u), float(v))
        # cv2's solver is double precision but not bit-exact - a millimeter
        # on a 100 m pitch is far below any meaningful calibration error.
        assert got_x == pytest.approx(px, abs=1e-3)
        assert got_y == pytest.approx(py, abs=1e-3)


def test_estimate_homography_more_than_four_points_uses_ransac():
    """Extra (noisy) correspondences are accepted and keep the fit sane."""
    rng = np.random.default_rng(0)
    image_pts = np.array(IMAGE_CORNERS + [(80.0, 60.0), (240.0, 180.0)])
    pitch_pts = np.array(PITCH_CORNERS + [(25.0, 17.0), (78.0, 50.0)])
    image_pts = image_pts + rng.normal(0.0, 0.5, image_pts.shape)

    transformer = PitchTransformer()
    transformer.estimate_homography(image_pts, pitch_pts)
    assert transformer.point_count == 6
    assert transformer.mean_error_meters < 1.0  # under a meter, well calibrated


def test_estimate_homography_requires_minimum_points():
    transformer = PitchTransformer(PitchConfig(min_points=4))
    with pytest.raises(ValueError, match="at least 4"):
        transformer.estimate_homography(IMAGE_CORNERS[:3], PITCH_CORNERS[:3])


def test_estimate_homography_rejects_mismatched_lengths():
    transformer = PitchTransformer()
    with pytest.raises(ValueError):
        transformer.estimate_homography(IMAGE_CORNERS, PITCH_CORNERS[:3])


def test_estimate_homography_rejects_collinear_points():
    transformer = PitchTransformer()
    line = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    with pytest.raises(ValueError, match="collinear"):
        transformer.estimate_homography(line, PITCH_CORNERS)


def test_estimate_homography_rejects_non_finite_points():
    transformer = PitchTransformer()
    bad = [(0.0, 0.0), (float("nan"), 0.0), (10.0, 5.0), (0.0, 5.0)]
    with pytest.raises(ValueError, match="finite"):
        transformer.estimate_homography(bad, PITCH_CORNERS)


# ---------------------------------------------------------------------- #
# image_to_pitch API
# ---------------------------------------------------------------------- #

def test_uncalibrated_transformer_raises():
    """image_to_pitch() must fail loudly before --calibrate was run."""
    transformer = PitchTransformer()
    assert not transformer.is_ready
    with pytest.raises(RuntimeError, match="not calibrated"):
        transformer.image_to_pitch(100.0, 200.0)
    with pytest.raises(RuntimeError):
        transformer.transform_point(100.0, 200.0)


def test_image_to_pitch_returns_meters_inside_pitch():
    transformer = make_transformer()
    # Centre of the frame maps into the pitch rectangle (in meters).
    pitch_x, pitch_y = transformer.image_to_pitch(FRAME_W / 2, FRAME_H / 2)
    assert 0.0 <= pitch_x <= 105.0
    assert 0.0 <= pitch_y <= 68.0
    assert isinstance(pitch_x, float) and isinstance(pitch_y, float)


def test_image_to_pitch_rejects_non_finite_input():
    transformer = make_transformer()
    with pytest.raises(ValueError, match="finite"):
        transformer.image_to_pitch(float("inf"), 10.0)


def test_transform_point_returns_coordinate():
    transformer = make_transformer()
    coord = transformer.transform_point(*foot_position((10.0, 20.0, 50.0, 120.0)))
    assert isinstance(coord, PitchCoordinate)
    expected = transformer.image_to_pitch(30.0, 120.0)
    assert (coord.x, coord.y) == pytest.approx(expected)


# ---------------------------------------------------------------------- #
# Calibration persistence
# ---------------------------------------------------------------------- #

def test_save_and_load_calibration_roundtrip(tmp_path: Path):
    path = tmp_path / "calib" / "pitch.json"
    original = make_transformer()
    saved = original.save_calibration(path, frame_index=17, image_size=(FRAME_W, FRAME_H))
    assert saved == path and path.exists()

    restored = PitchTransformer(PitchConfig(calibration_path=path))
    restored.load_calibration()
    assert restored.is_ready
    assert restored.frame_index == 17
    assert restored.image_size == (FRAME_W, FRAME_H)
    assert restored.point_count == 4

    # Same projections after reload.
    for u, v in [(0.0, 0.0), (160.0, 120.0), (300.0, 220.0)]:
        assert restored.image_to_pitch(u, v) == pytest.approx(
            original.image_to_pitch(u, v), abs=1e-9
        )


def test_load_calibration_missing_file(tmp_path: Path):
    transformer = PitchTransformer(PitchConfig(calibration_path=tmp_path / "none.json"))
    with pytest.raises(FileNotFoundError):
        transformer.load_calibration()


def test_load_calibration_dimension_mismatch(tmp_path: Path):
    path = tmp_path / "pitch.json"
    make_transformer().save_calibration(path)

    # Same file, but the config now claims a different pitch size.
    other = PitchTransformer(PitchConfig(length_meters=90.0, calibration_path=path))
    with pytest.raises(RuntimeError, match="align them"):
        other.load_calibration()


def test_save_without_calibration_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No calibration"):
        PitchTransformer().save_calibration(tmp_path / "x.json")


# ---------------------------------------------------------------------- #
# Top-down diagram
# ---------------------------------------------------------------------- #

def test_draw_pitch_shape_and_markings():
    diagram = draw_pitch(105.0, 68.0, (700, 450))
    assert diagram.shape == (450, 700, 3)
    white = (diagram[:, :, 0] > 200) & (diagram[:, :, 1] > 200) & (diagram[:, :, 2] > 200)
    assert white.sum() > 1000, "pitch markings should be drawn"
    # Halfway line: column at x = 52.5 m contains white pixels.
    halfway_col = int(round(52.5 / 105.0 * 700))
    assert white[:, halfway_col].any()


def test_draw_pitch_invalid_size():
    with pytest.raises(ValueError):
        draw_pitch(105.0, 68.0, (0, 100))
