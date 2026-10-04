"""
Stage 5: Side-by-Side Output Tests
==================================

Covers the ``--stage5`` composite (annotated footage | tactical pitch):
geometry (even width, writer/frame agreement, OpenCV readback), the Stage
6/7 analysis wiring, and the pitch-renderer drawing primitives (edges under
player dots, text overlay).
"""

from pathlib import Path

import numpy as np
import pytest

from app.pitch_renderer import (
    COLOR_TEAM_A,
    SEPARATOR_WIDTH,
    PitchRenderer,
    _PlayerRenderInfo,
    build_overlay_lines,
    draw_overlay,
    pitch_panel_width,
    render_side_by_side,
)
from app.pipeline import TacticalPipeline
from app.video import VideoReader

from test_pipeline import (
    FRAME_COUNT,
    FRAME_SIZE,
    BoxStubDetector,
    StubDetector,
    make_config,
    make_pitch_calibration,
    make_pitch_config,
    write_test_video,
    write_two_team_video,
    TWO_TEAM_BOXES,
    TWO_TEAM_FRAME_SIZE,
)

# The default renderer letterboxed to the 240px-tall test clips: 700*240/450
# = 373.3 -> 373, forced even -> 374. Composite: 320 + 4 + 374 = 698.
EXPECTED_PANEL_240 = pitch_panel_width(240, PitchRenderer())
EXPECTED_COMPOSITE_240 = FRAME_SIZE[0] + SEPARATOR_WIDTH + EXPECTED_PANEL_240


# ---------------------------------------------------------------------- #
# Geometry helpers
# ---------------------------------------------------------------------- #

def test_pitch_panel_width_is_always_even():
    for height in (144, 240, 360, 480, 720, 1080, 1440):
        width = pitch_panel_width(height, PitchRenderer())
        assert width > 0
        assert width % 2 == 0, f"odd panel width {width} for height {height}"


def test_pitch_panel_width_matches_the_renderers_own_aspect():
    """The value must be self-consistent for the renderer it is computed with."""
    base = PitchRenderer()
    panel = pitch_panel_width(720, base)
    renderer = PitchRenderer(rendered_size=(panel, 720))
    assert pitch_panel_width(720, renderer) == panel
    assert panel % 2 == 0


def test_render_side_by_side_dimensions():
    video = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (40, 90, 40), dtype=np.uint8)
    composite = render_side_by_side(video, PitchRenderer(), [], {})

    assert composite.shape == (FRAME_SIZE[1], EXPECTED_COMPOSITE_240, 3)
    assert EXPECTED_COMPOSITE_240 % 2 == 0

    # 4px grey separator right after the video half.
    separator = composite[:, FRAME_SIZE[0]:FRAME_SIZE[0] + SEPARATOR_WIDTH]
    assert np.all(np.abs(separator.astype(int) - 128) <= 2)

    # The video half is untouched (same pixels as the input frame).
    assert np.array_equal(composite[:, :FRAME_SIZE[0]], video)


# ---------------------------------------------------------------------- #
# End-to-end: --stage5 output
# ---------------------------------------------------------------------- #

def test_default_pipeline_width_is_unchanged(tmp_path: Path):
    """Without stage5 the classic single-pane output must stay as it was."""
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    output_path = tmp_path / "plain.mp4"

    TacticalPipeline(make_config(input_path, output_path), detector=StubDetector()).run()

    with VideoReader(output_path) as reader:
        assert reader.width == FRAME_SIZE[0]
        assert reader.height == FRAME_SIZE[1]


def test_stage5_writes_a_readable_even_width_composite(tmp_path: Path):
    """Composite survives the codec: written width == read-back width."""
    input_path = write_test_video(tmp_path / "input.mp4")
    output_path = tmp_path / "stage5.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector(), stage5=True
    ).run()

    assert stats.frames_processed == FRAME_COUNT
    assert output_path.exists()

    expected_width = FRAME_SIZE[0] + SEPARATOR_WIDTH + EXPECTED_PANEL_240
    assert expected_width % 2 == 0, "codec would truncate an odd width"

    with VideoReader(output_path) as reader:
        assert reader.width == expected_width
        assert reader.height == FRAME_SIZE[1]
        assert reader.total_frames == FRAME_COUNT
        _, frame = next(reader.read_frames())

    assert frame.shape == (FRAME_SIZE[1], expected_width, 3)
    # The right half is the pitch panel (grass green), the left stays video.
    pitch_panel = frame[:, FRAME_SIZE[0] + SEPARATOR_WIDTH:]
    grass = pitch_panel[..., 1].astype(int)
    assert (grass > 80).mean() > 0.5


def test_stage5_with_calibration_runs_stage6_and_stage7(tmp_path: Path):
    """Two-team clip + calibration -> formations and teammate graph are filled."""
    input_path = write_two_team_video(tmp_path / "two_teams.mp4")
    calibration = make_pitch_calibration(
        tmp_path / "calib.json", frame_size=TWO_TEAM_FRAME_SIZE
    )
    output_path = tmp_path / "stage7.mp4"
    config = make_pitch_config(input_path, output_path, calibration)

    stats = TacticalPipeline(
        config, detector=BoxStubDetector(TWO_TEAM_BOXES), stage5=True
    ).run()

    # Stage 6: per-team formation read-out from the last frame.
    assert set(stats.formations) == {"TEAM_A", "TEAM_B"}
    team_a = stats.formations["TEAM_A"]
    assert team_a["players"] == 2  # two committed players per kit
    assert team_a["formation"] == "UNKNOWN"  # 2 players: no three-line shape
    assert set(team_a) >= {"formation", "confidence", "width", "depth", "compactness"}

    # Stage 7: teammate graph from the last frame. Both pairs sit ~20 m apart,
    # comfortably inside the 25 m cap, so each team gets exactly one link.
    assert stats.graph_result is not None
    metrics = stats.graph_result.metrics
    for label in ("TEAM_A", "TEAM_B"):
        assert metrics[label]["players"] == 2
        assert metrics[label]["connections"] == 1
        assert metrics[label]["density"] == pytest.approx(1.0)
        assert metrics[label]["avg_teammate_distance"] > 0

    # The composite itself reads back at the exact written size.
    expected_width = TWO_TEAM_FRAME_SIZE[0] + SEPARATOR_WIDTH + EXPECTED_PANEL_240
    with VideoReader(output_path) as reader:
        assert reader.width == expected_width
        assert reader.height == TWO_TEAM_FRAME_SIZE[1]


def test_stage5_without_calibration_still_writes_the_composite(tmp_path: Path):
    """No calibration -> pitch panel empty + analysis off, but the run works."""
    input_path = write_test_video(tmp_path / "input.mp4", frames=5)
    output_path = tmp_path / "stage5.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector(), stage5=True
    ).run()

    assert stats.frames_processed == 5
    assert stats.formations == {}
    assert stats.graph_result is None
    with VideoReader(output_path) as reader:
        assert reader.width == EXPECTED_COMPOSITE_240


# ---------------------------------------------------------------------- #
# Renderer primitives
# ---------------------------------------------------------------------- #

def test_edges_are_drawn_under_the_player_dots():
    from app.pitch import PitchCoordinate
    from app.team_graph import GraphEdge

    players = [
        _PlayerRenderInfo(track_id=1, color=COLOR_TEAM_A, coordinate=PitchCoordinate(30.0, 34.0)),
        _PlayerRenderInfo(track_id=2, color=COLOR_TEAM_A, coordinate=PitchCoordinate(70.0, 34.0)),
    ]
    edge = GraphEdge(team="TEAM_A", id_a=1, id_b=2, distance=40.0)

    base = PitchRenderer().render(players)
    linked = PitchRenderer().render(players, edges=[edge])

    # Default renderer is 700x450; the segment midpoint (x=50m, y=34m) sits
    # between the two dots.
    mid = (int(round(50.0 / 105 * 700)), int(round((68 - 34) / 68 * 450)))
    mid_base, mid_linked = base[mid[1], mid[0]], linked[mid[1], mid[0]]

    def is_team_colour(pixel) -> bool:
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        return b > 150 and g < 120 and r < 120  # COLOR_TEAM_A = (255, 60, 0)

    assert not is_team_colour(mid_base), "no edge -> grass at the midpoint"
    assert is_team_colour(mid_linked), "edge missing between the two players"

    # The dot itself is drawn *after* the edge: its centre stays team colour
    # and its black outline ring survives.
    dot = (int(round(30.0 / 105 * 700)), int(round((68 - 34) / 68 * 450)))
    assert is_team_colour(linked[dot[1], dot[0]])


def test_edges_skipped_when_an_endpoint_is_missing():
    from app.pitch import PitchCoordinate
    from app.team_graph import GraphEdge

    players = [
        _PlayerRenderInfo(track_id=1, color=COLOR_TEAM_A, coordinate=PitchCoordinate(30.0, 34.0)),
    ]
    edge = GraphEdge(team="TEAM_A", id_a=1, id_b=2, distance=40.0)

    linked = PitchRenderer().render(players, edges=[edge])  # must not raise
    assert linked.shape == (450, 700, 3)


def test_build_overlay_lines_from_formations_and_graph():
    from app.team_graph import GraphEdge, TeamGraphResult

    formations = {
        "TEAM_A": {
            "team": "TEAM_A", "formation": "4-3-3", "confidence": 0.87,
            "width": 41.2, "depth": 33.5, "compactness": 0.09,
            "avg_teammate_distance": 11.4, "players": 11,
        },
        "TEAM_B": {
            "team": "TEAM_B", "formation": "UNKNOWN", "confidence": 0.1,
            "width": 0.0, "depth": 0.0, "compactness": 0.0,
            "avg_teammate_distance": 0.0, "players": 0,
        },
    }
    graph = TeamGraphResult(
        edges=[GraphEdge(team="TEAM_A", id_a=1, id_b=2, distance=12.0)],
        metrics={
            "TEAM_A": {
                "team": "TEAM_A", "players": 11, "connections": 14,
                "density": 0.25, "avg_teammate_distance": 11.4,
                "max_teammate_distance": 33.0,
            },
            "TEAM_B": {
                "team": "TEAM_B", "players": 0, "connections": 0,
                "density": 0.0, "avg_teammate_distance": 0.0,
                "max_teammate_distance": 0.0,
            },
        },
    )

    lines = build_overlay_lines(formations, graph)
    texts = [line if isinstance(line, str) else line[0] for line in lines]

    # One formation line + one network line for the populated team only.
    assert len(lines) == 2
    assert "4-3-3" in texts[0] and "TEAM_A" in texts[0]
    assert "87%" in texts[0]
    assert "14" in texts[1] and "0.25" in texts[1]
    assert any("33.0m" in text for text in texts)
    # TEAM_B has no visible players -> no lines.


def test_build_overlay_lines_is_empty_without_data():
    assert build_overlay_lines(None, None) == []
    assert build_overlay_lines({}, None) == []


def test_draw_overlay_stamps_text_and_darkens_the_box():
    panel = np.full((240, EXPECTED_PANEL_240, 3), (45, 120, 45), dtype=np.uint8)
    original = panel.copy()

    returned = draw_overlay(panel, [("TEAM_A 4-3-3", COLOR_TEAM_A), "TEAM_B UNKNOWN"])

    assert returned is panel
    # The translucent background box repaints a few thousand pixels darker.
    changed = int(np.any(panel != original, axis=2).sum())
    assert changed > 1500, "translucent background box missing"
    box = panel[6:45, 6:95]
    assert box.mean() < original[6:45, 6:95].mean() - 5

    # Coloured text pixels (b dominates, g/r low - grass and white text differ).
    b, g, r = panel[..., 0].astype(int), panel[..., 1].astype(int), panel[..., 2].astype(int)
    assert ((b > 100) & (b > g + 50) & (r < 160)).sum() > 10

    # Nothing drawn without lines.
    untouched = panel.copy()
    draw_overlay(untouched, [])
    assert np.array_equal(untouched, panel)


def test_video_writer_reports_open_state(tmp_path: Path):
    from app.video import VideoWriter

    path = tmp_path / "probe.mp4"
    with VideoWriter(path, fps=25.0, width=698, height=240) as writer:
        assert writer.isOpened() is True
        writer.write(np.zeros((240, 698, 3), dtype=np.uint8))
    assert writer.isOpened() is False  # released
