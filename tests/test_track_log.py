"""
Unit Tests for the Track Log
============================

Covers the per-frame tracking record store: the documented record structure,
frame queries, and CSV export / import.
"""

from pathlib import Path

from app.track_log import CSV_FIELDS, TrackLog, TrackRecord
from app.tracker import TrackedObject


def make_track(track_id: int = 7, bbox=(10.0, 20.0, 30.0, 60.0)) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        class_id=0,
        class_name="player",
        confidence=0.9,
    )


def test_record_from_track_uses_bbox_geometry():
    record = TrackRecord.from_track(frame=100, timestamp=4.04, track=make_track())

    assert record.frame == 100
    assert record.timestamp == 4.04
    assert record.player_id == 7
    assert record.bbox == (10.0, 20.0, 30.0, 60.0)
    assert record.center == (20.0, 40.0)
    assert (record.cx, record.cy) == (20.0, 40.0)


def test_record_to_dict_matches_documented_structure():
    data = TrackRecord.from_track(100, 4.04, make_track()).to_dict()

    assert list(data) == ["frame", "timestamp", "player_id", "bbox", "center"]
    assert data["frame"] == 100
    assert data["bbox"] == [10.0, 20.0, 30.0, 60.0]  # lists, as documented
    assert data["center"] == [20.0, 40.0]


def test_log_queries_frames_and_ids():
    log = TrackLog()
    log.add_track(0, 0.0, make_track(1))
    log.add_track(1, 0.04, make_track(1))
    log.add_track(1, 0.04, make_track(2))

    assert len(log) == 3
    assert log.unique_ids == {1, 2}
    assert log.frames == {0, 1}
    assert [r.player_id for r in log.for_frame(1)] == [1, 2]
    assert log.for_frame(42) == []
    assert len(log.to_dicts()) == 3


def test_csv_header_and_row_values(tmp_path: Path):
    log = TrackLog()
    log.add_track(0, 0.0, make_track(7))

    path = log.save_csv(tmp_path / "tracks.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert lines[0] == "frame,timestamp,player_id,x1,y1,x2,y2,cx,cy"
    assert lines[1] == "0,0.0,7,10.0,20.0,30.0,60.0,20.0,40.0"


def test_csv_export_creates_missing_directories_and_empty_files(tmp_path: Path):
    path = TrackLog().save_csv(tmp_path / "new" / "dir" / "tracks.csv")

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == ",".join(CSV_FIELDS)


def test_csv_round_trip_is_lossless(tmp_path: Path):
    log = TrackLog(
        [
            TrackRecord.from_track(3, 0.12, make_track(5)),
            TrackRecord.from_track(4, 0.16, make_track(6, bbox=(0.0, 1.0, 2.0, 3.0))),
        ]
    )

    reloaded = TrackLog.from_csv(log.save_csv(tmp_path / "round_trip.csv"))

    assert len(reloaded) == 2
    assert reloaded.records == log.records


def test_clear_and_repr():
    log = TrackLog()
    log.add_track(0, 0.0, make_track(1))
    assert "records=1" in repr(log)

    log.clear()
    assert len(log) == 0
    assert log.unique_ids == set()
