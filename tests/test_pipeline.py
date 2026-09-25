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

from app.config import AppConfig, VideoConfig
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
    )


def count_ball_marker_pixels(frame: np.ndarray) -> int:
    """Pixels close to the yellow ball marker colour (compression-tolerant)."""
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return int(((g > 170) & (r > 170) & (b < 110)).sum())


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
