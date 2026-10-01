"""
Unit Tests for the Visualization Module
=======================================

Covers the overlay primitives: markers keep their ID labels inside the frame,
and the HUD outline stays concentric with its text.
"""

import cv2
import numpy as np

import app.visualization as visualization
from app.pipeline import TacticalPipeline
from app.visualization import TacticalVisualizer


def extent(mask: np.ndarray) -> tuple[int, int]:
    """Leftmost/rightmost column where the mask is set."""
    xs = np.where(mask.any(axis=0))[0]
    return (int(xs.min()), int(xs.max())) if xs.size else (0, 0)


def test_foot_marker_draws_ellipse_and_label():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = TacticalVisualizer().draw_player_marker(frame, (100.0, 50.0, 200.0, 280.0), track_id=7)
    assert out is frame
    assert out.any(), "something should have been drawn"


def test_track_id_label_stays_inside_frame_when_feet_are_at_the_edge(monkeypatch):
    """A player cut off by the bottom edge must not lose their ID off-screen."""
    height, width = 200, 300
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    calls: list[tuple[str, tuple[int, int]]] = []
    real_put_text = cv2.putText

    def spy(image, text, org, *args, **kwargs):
        calls.append((text, org))
        return real_put_text(image, text, org, *args, **kwargs)

    monkeypatch.setattr(visualization.cv2, "putText", spy)

    # Bbox bottom sits exactly on the frame edge.
    TacticalVisualizer().draw_player_marker(frame, (100.0, 40.0, 200.0, float(height)), track_id=3)

    assert calls, "the ID label should have been drawn"
    for text, (x, y) in calls:
        assert text == "#3"
        assert 0 <= x < width, f"label x={x} outside frame width {width}"
        assert 0 <= y <= height, f"label y={y} outside frame height {height}"


def test_hud_outline_does_not_overshoot_the_text():
    """A thick black pass widens OpenCV's glyph advance - it must not be used."""
    height, width = 80, 500
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    TacticalPipeline._draw_hud(frame, 42, 5, 3, 7)

    light = frame.sum(axis=2) > 600
    dark = frame.sum(axis=2) < 100
    assert light.any() and dark.any(), "HUD should have white text on a dark outline"

    _, light_end = extent(light)
    _, dark_end = extent(dark)
    assert dark_end - light_end <= 3, (
        f"dark outline extends {dark_end - light_end}px past the text "
        f"(expected a concentric ~1px fringe)"
    )


# ---------------------------------------------------------------------- #
# Stage 2: bounding box + confidence annotation
# ---------------------------------------------------------------------- #

def _spy_put_text(monkeypatch) -> list[tuple[str, tuple[int, int]]]:
    """Record every (text, origin) passed to cv2.putText."""
    calls: list[tuple[str, tuple[int, int]]] = []
    real_put_text = cv2.putText

    def spy(image, text, org, *args, **kwargs):
        calls.append((text, org))
        return real_put_text(image, text, org, *args, **kwargs)

    monkeypatch.setattr(visualization.cv2, "putText", spy)
    return calls


def test_player_marker_draws_the_bounding_box():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    TacticalVisualizer().draw_player_marker(frame, (100.0, 50.0, 200.0, 280.0), track_id=7)

    assert frame[50, 150].any(), "top edge of the bbox should be drawn"
    assert frame[165, 100].any(), "left edge of the bbox should be drawn"
    assert frame[280, 200].any(), "right edge of the bbox should be drawn"


def test_player_marker_label_shows_id_and_confidence(monkeypatch):
    calls = _spy_put_text(monkeypatch)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    TacticalVisualizer().draw_player_marker(
        frame, (100.0, 50.0, 200.0, 280.0), track_id=7, confidence=0.87
    )

    texts = {text for text, _ in calls}
    assert "#7 0.87" in texts, f"expected '#7 0.87', drew {texts}"


def test_player_marker_label_is_inside_the_frame_when_head_is_cut_off(monkeypatch):
    """A player whose head touches the top edge must not lose their ID off-screen."""
    height, width = 200, 300
    calls = _spy_put_text(monkeypatch)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    TacticalVisualizer().draw_player_marker(frame, (100.0, 0.0, 200.0, 160.0), track_id=3)

    assert calls, "the ID label should have been drawn"
    for text, (x, y) in calls:
        assert text.startswith("#3")
        assert 0 <= x < width, f"label x={x} outside frame width {width}"
        assert 0 <= y <= height, f"label y={y} outside frame height {height}"


def test_player_marker_can_skip_the_bounding_box():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    TacticalVisualizer().draw_player_marker(
        frame, (100.0, 50.0, 200.0, 280.0), track_id=7, draw_box=False
    )
    assert not frame[50, 150].any(), "no box should have been drawn"
    assert frame.any(), "the ellipse/label should still be drawn"


# ---------------------------------------------------------------------- #
# Stage 3: team legend
# ---------------------------------------------------------------------- #

def _near(frame: np.ndarray, bgr: tuple[int, int, int], tol: int = 6) -> int:
    b, g, r = frame[..., 0].astype(int), frame[..., 1].astype(int), frame[..., 2].astype(int)
    return int(((abs(b - bgr[0]) < tol) & (abs(g - bgr[1]) < tol) & (abs(r - bgr[2]) < tol)).sum())


def test_team_legend_draws_both_team_swatch_colours_top_right():
    frame = np.zeros((240, 480, 3), dtype=np.uint8)
    counts = {"TEAM_A": 5, "TEAM_B": 3, "UNKNOWN": 1}

    out = TacticalVisualizer().draw_team_legend(frame, counts)

    assert out is frame
    # Swatches live in the top-right panel only.
    panel = frame[:100, 480 - 140:]
    assert _near(panel, visualization.COLOR_TEAM_A) > 100, "TEAM_A swatch missing"
    assert _near(panel, visualization.COLOR_TEAM_B) > 100, "TEAM_B swatch missing"
    assert _near(panel, visualization.COLOR_UNKNOWN) > 100, "UNKNOWN swatch missing"
    # Outside the panel no legend colours were sprayed.
    rest = frame.copy()
    rest[:100, 480 - 140:] = 0
    assert _near(rest, visualization.COLOR_TEAM_A) == 0
    assert _near(rest, visualization.COLOR_TEAM_B) == 0


def test_team_legend_works_without_counts():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    TacticalVisualizer().draw_team_legend(frame)
    assert frame.any(), "legend should be drawn even without counts"


# ---------------------------------------------------------------------- #
# Stage 4: top-down radar minimap
# ---------------------------------------------------------------------- #

def color_hits(img: np.ndarray, bgr: tuple[int, int, int], tol: int = 6) -> int:
    """Pixels close to an exact BGR colour."""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    return int(
        ((abs(b - bgr[0]) < tol) & (abs(g - bgr[1]) < tol) & (abs(r - bgr[2]) < tol)).sum()
    )


def test_radar_minimap_draws_pitch_panel_and_player_dots():
    from app.pitch import PitchCoordinate
    from app.visualization import COLOR_BALL, COLOR_TEAM_A, COLOR_TEAM_B

    frame = np.full((240, 320, 3), (40, 90, 40), dtype=np.uint8)
    visualizer = TacticalVisualizer(pitch_radar_size=(300, 200), pitch_size=(105.0, 68.0))
    players = [
        (PitchCoordinate(10.0, 50.0), COLOR_TEAM_A, 7),
        (PitchCoordinate(90.0, 20.0), COLOR_TEAM_B, 9),
    ]

    out = visualizer.draw_radar_minimap(frame, players, PitchCoordinate(52.5, 34.0))

    assert out is frame, "the radar is an extra panel on the same frame"
    panel = out[32:232, 8:308]  # 300x200 radar, bottom-left, 8 px margin
    # Diagram grass (45, 120, 45) is distinct from the video green (40, 90, 40).
    assert color_hits(panel, (45, 120, 45), tol=10) > 5000, "pitch panel missing"
    assert color_hits(panel, COLOR_TEAM_A, tol=6) > 0, "Team A dot missing"
    assert color_hits(panel, COLOR_TEAM_B, tol=6) > 0, "Team B dot missing"
    assert color_hits(panel, COLOR_BALL, tol=6) > 0, "ball dot missing"


def test_radar_minimap_fits_small_frames():
    from app.pitch import PitchCoordinate

    frame = np.full((100, 200, 3), (40, 90, 40), dtype=np.uint8)
    visualizer = TacticalVisualizer(pitch_radar_size=(300, 200))
    out = visualizer.draw_radar_minimap(
        frame, [(PitchCoordinate(50.0, 30.0), (255, 60, 0), 1)]
    )
    # The panel is scaled down instead of falling off the tiny frame.
    assert color_hits(out, (45, 120, 45), tol=10) > 200


def test_radar_minimap_matches_camera_orientation():
    """y=0 (near sideline) renders at the BOTTOM of the panel.

    The camera sits on the near side, so in the video the near sideline is at
    the bottom of the frame; the radar mirrors that view instead of drawing
    +y downward like a plain plot (which would vertically flip formations
    against the footage).
    """
    from app.pitch import PitchCoordinate
    from app.visualization import COLOR_TEAM_A, COLOR_TEAM_B

    frame = np.full((240, 320, 3), (40, 90, 40), dtype=np.uint8)
    visualizer = TacticalVisualizer(pitch_radar_size=(300, 200), pitch_size=(105.0, 68.0))
    visualizer.draw_radar_minimap(
        frame,
        [
            (PitchCoordinate(52.5, 4.0), COLOR_TEAM_A, 1),   # near the camera
            (PitchCoordinate(52.5, 64.0), COLOR_TEAM_B, 2),  # far side
        ],
    )
    panel = frame[32:232, 8:308]

    def mean_row(color, tol=6):
        b, g, r = panel[:, :, 0].astype(int), panel[:, :, 1].astype(int), panel[:, :, 2].astype(int)
        mask = (abs(b - color[0]) < tol) & (abs(g - color[1]) < tol) & (abs(r - color[2]) < tol)
        rows = np.nonzero(mask)[0]
        assert rows.size > 0, f"dot colour {color} missing from panel"
        return rows.mean()

    near_row, far_row = mean_row(COLOR_TEAM_A), mean_row(COLOR_TEAM_B)
    # Panel is 200 px for 68 m: y=4 must sit far below y=64.
    assert near_row > far_row + 50, (
        f"near-side dot (row {near_row:.0f}) must render below far-side dot (row {far_row:.0f})"
    )


def test_radar_minimap_works_without_players():
    frame = np.full((240, 320, 3), (40, 90, 40), dtype=np.uint8)
    out = TacticalVisualizer().draw_radar_minimap(frame, [])
    assert out is frame
    assert color_hits(out[32:232, 8:308], (45, 120, 45), tol=10) > 5000
