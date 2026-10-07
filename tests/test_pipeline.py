"""
End-to-End Pipeline Tests
=========================

Feeds a small synthetic video through TacticalPipeline with a stub detector, so
the orchestration (reader -> detect -> track -> annotate -> writer) is covered
without downloading weights or needing real footage.
"""

from pathlib import Path
from typing import List

import csv

import cv2
import numpy as np
import pytest

from app.config import AppConfig, PitchConfig, VideoConfig
from app.detector import Detection
from app.pipeline import TacticalPipeline
from app.track_log import TrackLog
from app.video import VideoReader

FRAME_COUNT = 30
FRAME_SIZE = (320, 240)  # (width, height)
FPS = 25.0


class StubDetector:
    """Deterministic detector: reports the moving white rectangle it is given."""

    def __init__(self, with_ball: bool = False, empty: bool = False):
        self.with_ball = with_ball
        self.empty = empty
        self.step = 0
        self.load_count = 0

    def load_model(self) -> None:
        self.load_count += 1

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self.step += 1
        if self.empty:
            return []
        x = 10 + (self.step - 1) * 3
        detections = [
            Detection(
                bbox=(float(x), 80.0, float(x + 40), 180.0),
                confidence=0.9,
                class_id=0,
                class_name="player",
            )
        ]
        if self.with_ball:
            detections.append(
                Detection(
                    bbox=(float(x + 45), 90.0, float(x + 65), 110.0),
                    confidence=0.8,
                    class_id=32,
                    class_name="ball",
                )
            )
        return detections


def write_test_video(path: Path, frames: int = FRAME_COUNT) -> Path:
    """Render a short clip containing a moving white rectangle on green."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE,
    )
    assert writer.isOpened(), f"Could not open test video writer for {path}"
    for i in range(frames):
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (40, 90, 40), dtype=np.uint8)
        x = 10 + i * 3
        cv2.rectangle(frame, (x, 80), (x + 40, 180), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def make_config(input_path: Path, output_path: Path, **video_kwargs) -> AppConfig:
    return AppConfig(
        video=VideoConfig(input_path=input_path, output_path=output_path, **video_kwargs),
        # Keep Stage 4 hermetic: never pick up a real calibration/pitch.json
        # that happens to exist in the working directory (the radar inset
        # would cover parts of the small synthetic test frame).
        pitch=PitchConfig(calibration_path=output_path.parent / "no_calibration.json"),
    )


def count_ball_marker_pixels(frame: np.ndarray) -> int:
    """Pixels close to the yellow ball marker colour (compression-tolerant)."""
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return int(((g > 170) & (r > 170) & (b < 110)).sum())


def count_color_pixels(frame: np.ndarray, bgr: tuple[int, int, int], tol: int = 25) -> int:
    """Pixels close to an exact BGR colour (marker / legend colours)."""
    b, g, r = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)
    return int(((abs(b - bgr[0]) < tol) & (abs(g - bgr[1]) < tol) & (abs(r - bgr[2]) < tol)).sum())


def test_pipeline_writes_annotated_video(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4")
    output_path = tmp_path / "annotated.mp4"
    stub = StubDetector()

    stats = TacticalPipeline(make_config(input_path, output_path), detector=stub).run()

    assert stub.load_count == 1, "the model should be loaded exactly once"
    assert stats.frames_processed == FRAME_COUNT
    assert output_path.exists() and output_path.stat().st_size > 0
    assert stats.unique_track_ids == {1}
    assert stats.players_last_frame == 1
    assert stats.mean_players_per_frame == pytest.approx(1.0)
    assert stats.processing_fps > 0

    with VideoReader(output_path) as reader:
        written = sum(1 for _ in reader.read_frames())
    assert written == FRAME_COUNT


def test_pipeline_handles_frames_without_detections(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector(empty=True)
    ).run()

    assert stats.frames_processed == 10
    assert stats.detections_total == 0
    assert stats.unique_track_ids == set()


def test_pipeline_draws_players_and_ball(tmp_path: Path):
    """tracker.player_only (default) tracks players only - the ball is drawn from
    the raw detection instead of getting a flickering ID of its own."""
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector(with_ball=True)
    ).run()

    # Stage 1 still sees both objects on every frame...
    assert stats.detections_total == 2 * 10
    # ...but only the player receives a track ID.
    assert stats.unique_track_ids == {1}
    assert stats.unique_player_ids == {1}
    assert output_path.exists()

    # The ball must still be visible in the annotated output.
    with VideoReader(output_path) as reader:
        _, first_frame = next(reader.read_frames())
    assert count_ball_marker_pixels(first_frame) > 10, "ball marker missing from output"


def test_pipeline_tracks_the_ball_when_player_only_is_disabled(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    config = make_config(input_path, tmp_path / "out.mp4")
    config.tracker.player_only = False

    stats = TacticalPipeline(config, detector=StubDetector(with_ball=True)).run()

    assert stats.unique_track_ids == {1, 2}
    assert stats.unique_player_ids == {1}


def test_pipeline_respects_stride_and_max_frames(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4")
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path, frame_stride=2, max_frames=5),
        detector=StubDetector(),
    ).run()

    assert stats.frames_processed == 5

    with VideoReader(output_path) as reader:
        assert reader.total_frames == 5


def test_pipeline_reports_missing_input(tmp_path: Path):
    missing = tmp_path / "nope.mp4"
    pipeline = TacticalPipeline(
        make_config(missing, tmp_path / "out.mp4"), detector=StubDetector()
    )
    with pytest.raises(FileNotFoundError):
        pipeline.run()


# ---------------------------------------------------------------------- #
# Stage 2: per-frame tracking records + CSV export
# ---------------------------------------------------------------------- #

def test_pipeline_records_every_tracked_player_per_frame(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4")
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector()
    ).run()

    log = stats.track_log
    # One record per frame for the single tracked player.
    assert len(log) == FRAME_COUNT
    assert log.unique_ids == {1}
    assert log.frames == set(range(FRAME_COUNT))

    first = log.for_frame(0)
    assert len(first) == 1
    record = first[0]
    assert record.player_id == 1
    assert record.bbox == (10.0, 80.0, 50.0, 180.0)
    assert record.center == (30.0, 130.0)
    # Source-frame timestamp: frame_index / fps (stride independent).
    assert record.timestamp == pytest.approx(0.0)
    assert log.for_frame(5)[0].timestamp == pytest.approx(5 / FPS)

    # Structure required by Stage 2.
    as_dict = record.to_dict()
    assert set(as_dict) == {"frame", "timestamp", "player_id", "bbox", "center"}
    assert as_dict["bbox"] == [10.0, 80.0, 50.0, 180.0]
    assert as_dict["center"] == [30.0, 130.0]


def test_pipeline_track_records_follow_frame_stride(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4")
    config = make_config(input_path, tmp_path / "out.mp4", frame_stride=2, max_frames=5)

    stats = TacticalPipeline(config, detector=StubDetector()).run()

    # Source frames 0, 2, 4, 6, 8 -> timestamps 0, 0.08, 0.16, ...
    assert [r.frame for r in stats.track_log.records] == [0, 2, 4, 6, 8]
    assert stats.track_log.records[1].timestamp == pytest.approx(2 / FPS)


def test_pipeline_exports_tracking_csv(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector()
    ).run()

    csv_path = stats.track_log.save_csv(tmp_path / "nested" / "tracks.csv")
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "frame",
        "timestamp",
        "player_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
    ]
    assert len(rows) == 1 + 10
    assert rows[1][0] == "0"
    assert rows[1][2] == "1"
    assert float(rows[1][1]) == pytest.approx(0.0)
    assert float(rows[1][3]) == pytest.approx(10.0)
    assert float(rows[1][7]) == pytest.approx(30.0)
    assert float(rows[1][8]) == pytest.approx(130.0)

    # The CSV can be read back losslessly.
    reloaded = TrackLog.from_csv(csv_path)
    assert len(reloaded) == 10
    assert reloaded.records[0].center == pytest.approx((30.0, 130.0))


# ---------------------------------------------------------------------- #
# Stage 3: team classification end-to-end
# ---------------------------------------------------------------------- #

# Kits in the synthetic footage - deliberately *not* the annotation colours.
KIT_A_BGR = (200, 70, 40)   # blue-ish
KIT_B_BGR = (40, 45, 200)   # red-ish

TWO_TEAM_BOXES = [
    (40.0, 60.0, 90.0, 200.0),     # kit A
    (130.0, 60.0, 180.0, 200.0),   # kit A
    (300.0, 60.0, 350.0, 200.0),   # kit B
    (390.0, 60.0, 440.0, 200.0),   # kit B
]
TWO_TEAM_FRAMES = 40


class BoxStubDetector:
    """Deterministic detector: always reports the four fixed boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.load_count = 0

    def load_model(self) -> None:
        self.load_count += 1

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return [
            Detection(bbox=b, confidence=0.9, class_id=0, class_name="player")
            for b in self.boxes
        ]


# The two-team boxes span x=40..440, so this clip needs a wider frame than the
# single-player default: with FRAME_SIZE the kit-B team fell outside the image
# and those players could never yield a jersey colour.
TWO_TEAM_FRAME_SIZE = (480, 240)

# Frame from which teams are considered committed: the clustering round needs
# min_cluster_samples across min_cluster_tracks tracks, then the assignment
# must hold for min_consistent_frames frames (0-based).
TEAM_SETTLE_FRAME = 20


def write_two_team_video(path: Path, frames: int = TWO_TEAM_FRAMES) -> Path:
    """Static four-player clip: two players per kit on a grass background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, TWO_TEAM_FRAME_SIZE
    )
    assert writer.isOpened(), f"Could not open test video writer for {path}"
    for _ in range(frames):
        frame = np.full(
            (TWO_TEAM_FRAME_SIZE[1], TWO_TEAM_FRAME_SIZE[0], 3), (45, 110, 45), dtype=np.uint8
        )
        for index, (x1, y1, x2, y2) in enumerate(TWO_TEAM_BOXES):
            color = KIT_A_BGR if index < 2 else KIT_B_BGR
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        writer.write(frame)
    writer.release()
    return path


def test_pipeline_classifies_two_teams_end_to_end(tmp_path: Path):
    """Raw clip in -> per-frame team records + per-team colours + legend out."""
    input_path = write_two_team_video(tmp_path / "two_teams.mp4")
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=BoxStubDetector(TWO_TEAM_BOXES)
    ).run()

    # One team record per tracked player per frame.
    assert len(stats.team_log) == 4 * TWO_TEAM_FRAMES
    assert stats.team_log.frames == set(range(TWO_TEAM_FRAMES))

    # Every player is classified (no UNKNOWN) once the model has settled.
    final_frame = TWO_TEAM_FRAMES - 1
    teams = {r.player_id: r.team for r in stats.team_log.for_frame(final_frame)}
    assert set(teams.values()) <= {"TEAM_A", "TEAM_B"}

    # Over the whole run each of the four players ends up on exactly one team:
    # two unique players per label (a mid-run switch would push the sum past 4)
    # and nobody stays UNKNOWN - only the warm-up frames before the first
    # commitment are labelled UNKNOWN.
    counts = stats.team_log.team_counts
    assert counts["TEAM_A"] == 2 and counts["TEAM_B"] == 2, counts
    ever_labeled = {r.player_id for r in stats.team_log if r.team in ("TEAM_A", "TEAM_B")}
    assert ever_labeled == {r.player_id for r in stats.team_log}, counts

    # The split follows the *kits*, not the geometry: the two left boxes share
    # one team and the two right boxes the other (which is which is decided
    # by the clustering).
    centers = {r.player_id: r.cx for r in stats.track_log.for_frame(final_frame)}
    left = [pid for pid, cx in centers.items() if cx < 250]
    right = [pid for pid, cx in centers.items() if cx >= 250]
    assert len(left) == 2 and len(right) == 2
    assert len({teams[pid] for pid in left}) == 1, f"kit A split: {teams}"
    assert len({teams[pid] for pid in right}) == 1, f"kit B split: {teams}"
    assert teams[left[0]] != teams[right[0]], "the two kits must differ"


def test_pipeline_writes_team_csv_with_documented_columns(tmp_path: Path):
    input_path = write_two_team_video(tmp_path / "two_teams.mp4")
    stats = TacticalPipeline(
        make_config(input_path, tmp_path / "out.mp4"), detector=BoxStubDetector(TWO_TEAM_BOXES)
    ).run()

    csv_path = stats.team_log.save_csv(tmp_path / "nested" / "teams.csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["frame", "timestamp", "player_id", "team", "cx", "cy"]
    assert len(rows) == 1 + 4 * TWO_TEAM_FRAMES
    assert {row[3] for row in rows[1:]} <= {"TEAM_A", "TEAM_B", "UNKNOWN"}
    assert float(rows[1][1]) == pytest.approx(0.0)
    # Rows line up with the Stage 2 tracking CSV (same frame, same centre).
    track_rows = {r.player_id: r for r in stats.track_log.for_frame(0)}
    for row in rows[1:6]:
        record = track_rows[int(row[2])]
        assert float(row[4]) == pytest.approx(record.cx)
        assert float(row[5]) == pytest.approx(record.cy)


def test_pipeline_draws_per_team_colours_and_legend(tmp_path: Path):
    """Team A / Team B players are annotated in different colours + legend."""
    input_path = write_two_team_video(tmp_path / "two_teams.mp4")
    output_path = tmp_path / "annotated.mp4"

    TacticalPipeline(
        make_config(input_path, output_path), detector=BoxStubDetector(TWO_TEAM_BOXES)
    ).run()

    from app.visualization import COLOR_TEAM_A, COLOR_TEAM_B

    # Teams are only coloured in once they are committed (cluster fit +
    # min_consistent_frames), so inspect a settled frame, not frame 0.
    with VideoReader(output_path) as reader:
        frames = reader.read_frames()
        for _ in range(TEAM_SETTLE_FRAME):
            _, frame = next(frames)

    # Markers + legend swatches use both team colours, the kits themselves do
    # not, so any hit above comes from the annotations.
    assert count_color_pixels(frame, COLOR_TEAM_A) > 50, "Team A colour missing"
    assert count_color_pixels(frame, COLOR_TEAM_B) > 50, "Team B colour missing"

    # The legend lives in the top-right corner.
    legend = frame[:100, frame.shape[1] - 140:]
    assert count_color_pixels(legend, COLOR_TEAM_A) > 50
    assert count_color_pixels(legend, COLOR_TEAM_B) > 50


# ---------------------------------------------------------------------- #
# Stage 4: pitch mapping end-to-end
# ---------------------------------------------------------------------- #

def make_pitch_calibration(path: Path, frame_size=FRAME_SIZE) -> Path:
    """Frame corners mapped onto the corners of a 105x68 pitch (meters)."""
    from app.config import PitchConfig
    from app.pitch import PitchTransformer

    width, height = frame_size
    transformer = PitchTransformer(PitchConfig(calibration_path=path))
    transformer.estimate_homography(
        [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
        [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)],
    )
    transformer.save_calibration(path, frame_index=0, image_size=frame_size)
    return path


def make_pitch_config(
    input_path: Path, output_path: Path, calibration_path: Path
) -> AppConfig:
    from app.config import PitchConfig

    return AppConfig(
        video=VideoConfig(input_path=input_path, output_path=output_path),
        pitch=PitchConfig(calibration_path=calibration_path),
    )


def test_pitch_stage_disabled_without_calibration(tmp_path: Path):
    """No calibration file -> the stage stays off and no pitch rows appear."""
    input_path = write_test_video(tmp_path / "input.mp4", frames=5)
    config = make_pitch_config(
        input_path, tmp_path / "out.mp4", tmp_path / "missing.json"
    )

    stats = TacticalPipeline(config, detector=StubDetector()).run()

    assert len(stats.pitch_log) == 0
    assert not stats.pitch_log.records


def test_pipeline_projects_foot_positions_in_meters(tmp_path: Path):
    """Every player frame gets a pitch coordinate in meters (foot position)."""
    input_path = write_test_video(tmp_path / "input.mp4")
    calibration = make_pitch_calibration(tmp_path / "calib.json")
    config = make_pitch_config(input_path, tmp_path / "out.mp4", calibration)

    stats = TacticalPipeline(config, detector=StubDetector()).run()

    assert len(stats.pitch_log) == FRAME_COUNT  # one player in every frame
    record = stats.pitch_log.records[0]

    # The homography is a pure scale between the two rectangles:
    # foot = centre/bottom of bbox (10, 80, 50, 180) -> (30, 180) px.
    assert record.pitch_x == pytest.approx(30.0 * 105.0 / FRAME_SIZE[0])
    assert record.pitch_y == pytest.approx(180.0 * 68.0 / FRAME_SIZE[1])
    assert 0.0 <= record.pitch_x <= 105.0
    assert 0.0 <= record.pitch_y <= 68.0
    assert record.team == "UNKNOWN"  # single player: no cluster fit yet
    assert stats.pitch_log.unique_ids == {1}


def test_pipeline_exports_pitch_csv(tmp_path: Path):
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    calibration = make_pitch_calibration(tmp_path / "calib.json")
    config = make_pitch_config(input_path, tmp_path / "out.mp4", calibration)

    stats = TacticalPipeline(config, detector=StubDetector()).run()
    csv_path = stats.pitch_log.save_csv(tmp_path / "nested" / "pitch.csv")

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "frame", "timestamp", "player_id", "team", "pitch_x", "pitch_y",
    ]
    assert len(rows) == 1 + 10
    assert rows[1][0] == "0" and rows[1][2] == "1"
    assert float(rows[1][4]) == pytest.approx(30.0 * 105.0 / FRAME_SIZE[0])
    assert float(rows[1][5]) == pytest.approx(180.0 * 68.0 / FRAME_SIZE[1])


def test_radar_inset_drawn_only_when_calibrated(tmp_path: Path):
    """Calibrated -> radar panel with team-coloured dots; otherwise untouched."""
    from app.tracker import TrackedObject
    from app.visualization import COLOR_TEAM_A

    tracks = [
        TrackedObject(
            track_id=1, bbox=(10.0, 80.0, 50.0, 180.0), class_id=0, class_name="player"
        )
    ]
    blank = (40, 90, 40)
    radar_grass = (45, 120, 45)

    # Without calibration: bottom-left stays plain video green.
    uncalibrated = TacticalPipeline(
        make_pitch_config(
            write_test_video(tmp_path / "in.mp4", frames=1),
            tmp_path / "out.mp4",
            tmp_path / "missing.json",
        ),
        detector=StubDetector(),
    )
    uncalibrated._probe_optional_stages()
    assert not uncalibrated._pitch_stage_enabled
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), blank, dtype=np.uint8)
    plain = uncalibrated._annotate(frame.copy(), [], tracks, 0)
    assert count_color_pixels(plain[32:232, 8:308], radar_grass, tol=10) == 0

    # With calibration: the 300x200 radar panel appears bottom-left with the
    # player dot in its team colour.
    calibrated = TacticalPipeline(
        make_pitch_config(
            write_test_video(tmp_path / "in2.mp4", frames=1),
            tmp_path / "out2.mp4",
            make_pitch_calibration(tmp_path / "calib.json"),
        ),
        detector=StubDetector(),
    )
    calibrated._probe_optional_stages()
    assert calibrated._pitch_stage_enabled
    calibrated.team_by_track[1] = 0  # pretend Stage 3 committed this player
    calibrated._update_pitch(tracks)
    assert 1 in calibrated.pitch_by_track

    annotated = calibrated._annotate(frame.copy(), [], tracks, 0)
    panel = annotated[32:232, 8:308]
    assert count_color_pixels(panel, radar_grass, tol=10) > 5000, "radar missing"
    assert count_color_pixels(panel, COLOR_TEAM_A, tol=6) > 0, "player dot missing"


# ---------------------------------------------------------------------- #
# Run diagnostics (detection / tracking / team attribution)
# ---------------------------------------------------------------------- #

def test_pipeline_fills_run_diagnostics(tmp_path: Path):
    """stats.diagnostics carries the full per-run report."""
    input_path = write_test_video(tmp_path / "input.mp4")
    pipeline = TacticalPipeline(
        make_config(input_path, tmp_path / "out.mp4"), detector=StubDetector()
    )
    stats = pipeline.run()

    diag = stats.diagnostics
    assert diag["frames"] == FRAME_COUNT
    assert diag["tracks"]["unique"] == 1
    assert diag["tracks"]["losses"] == 0          # the track lives to the end
    assert diag["detections"]["mean"] == pytest.approx(1.0)
    assert diag["detections"]["display_mean"] == pytest.approx(1.0)  # stub conf 0.9
    assert diag["active_tracks"]["max"] == 1
    # Single player: no two-team model -> nobody ever labelled.
    assert diag["teams"]["final_UNKNOWN"] == 1
    assert diag["teams"]["ab_switches_total"] == 0

    # The same object is exposed for programmatic access after the run.
    assert pipeline.diagnostics.summary()["frames"] == FRAME_COUNT


def test_pipeline_diagnostics_count_team_switches_in_a_two_team_run(tmp_path: Path):
    """A settled two-team clip should finish with zero A<->B switches."""
    input_path = write_two_team_video(tmp_path / "two_teams.mp4")
    stats = TacticalPipeline(
        make_config(input_path, tmp_path / "out.mp4"),
        detector=BoxStubDetector(TWO_TEAM_BOXES),
    ).run()

    teams = stats.diagnostics["teams"]
    assert stats.diagnostics["frames"] == TWO_TEAM_FRAMES
    assert teams["ab_switches_total"] == 0, teams
    assert teams["stable_TEAM_A"] + teams["stable_TEAM_B"] == 4, teams
    assert teams["final_UNKNOWN"] == 0, teams


# ---------------------------------------------------------------------- #
# --debug overlay
# ---------------------------------------------------------------------- #

# Badge background drawn behind the debug team/confidence text. Non-neutral on
# purpose: white-on-black text anti-aliasing produces equal-channel greys, so
# the badge colour must differ in *shape* (b != r) from any blend of text,
# grass, legend and marker colours.
DEBUG_BADGE_BGR = (50, 100, 150)


def test_debug_overlay_adds_badge_and_roi_only_when_enabled():
    """--debug adds the team/confidence badge + jersey ROI; default stays clean."""
    from app.team_classifier import TeamClassifier
    from app.tracker import TrackedObject

    config = make_config(Path("unused-in.mp4"), Path("unused-out.mp4"))
    tracks = [
        TrackedObject(
            track_id=1, bbox=(10.0, 80.0, 50.0, 180.0), class_id=0, class_name="player"
        )
    ]
    grass = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (40, 90, 40), dtype=np.uint8)

    debugged_pipeline = TacticalPipeline(config, detector=StubDetector(), debug=True)
    debugged_pipeline._team_stage_enabled = True
    debugged_pipeline._team_classifier = TeamClassifier(config.team_classifier)
    debugged_pipeline.team_by_track[1] = 0
    annotated = debugged_pipeline._annotate(grass.copy(), [], tracks, 0)
    assert count_color_pixels(annotated, DEBUG_BADGE_BGR, tol=6) > 50, "badge missing"

    # Same pipeline, debug off: the badge may not appear.
    debugged_pipeline.debug = False
    plain = debugged_pipeline._annotate(grass.copy(), [], tracks, 0)
    assert count_color_pixels(plain, DEBUG_BADGE_BGR, tol=6) == 0, "badge leaked"


def test_pipeline_runs_end_to_end_with_debug_enabled(tmp_path: Path):
    """The debug overlay must not disturb the normal run/write path."""
    input_path = write_test_video(tmp_path / "input.mp4", frames=10)
    output_path = tmp_path / "annotated.mp4"

    stats = TacticalPipeline(
        make_config(input_path, output_path), detector=StubDetector(), debug=True
    ).run()

    assert stats.frames_processed == 10
    assert output_path.exists() and output_path.stat().st_size > 0


# ---------------------------------------------------------------------- #
# Detection-for-display vs detection-for-tracking
# ---------------------------------------------------------------------- #

def test_ball_marker_is_gated_at_the_display_threshold():
    """Weak ball candidates stay tracker-only; the marker needs display quality."""
    from app.visualization import COLOR_BALL

    config = make_config(Path("unused-in.mp4"), Path("unused-out.mp4"))
    pipeline = TacticalPipeline(config, detector=StubDetector())
    grass = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (40, 90, 40), dtype=np.uint8)
    ball = Detection(
        bbox=(60.0, 90.0, 80.0, 110.0), confidence=0.2, class_id=32, class_name="ball"
    )

    weak = pipeline._annotate(grass.copy(), [ball], [], 0)
    assert count_color_pixels(weak, COLOR_BALL, tol=20) == 0, "weak ball leaked into the frame"

    ball.confidence = 0.9
    strong = pipeline._annotate(grass.copy(), [ball], [], 0)
    assert count_color_pixels(strong, COLOR_BALL, tol=20) > 0, "display-grade ball not drawn"
