"""
Unit Tests for the Stage 4 Pitch Log
====================================
Verifies the pitch-record schema and the --pitch-csv export contract:

    frame,timestamp,player_id,team,pitch_x,pitch_y
"""

import csv
from pathlib import Path

import pytest

from app.pitch import PitchCoordinate
from app.pitch_log import CSV_FIELDS, PitchLog, PitchRecord
from app.tracker import TrackedObject


def make_track(track_id: int = 7) -> TrackedObject:
    return TrackedObject(
        track_id=track_id, bbox=(10.0, 20.0, 50.0, 120.0), class_id=0, class_name="player"
    )


def test_csv_fields_contract():
    assert CSV_FIELDS == (
        "frame",
        "timestamp",
        "player_id",
        "team",
        "pitch_x",
        "pitch_y",
    )


def test_add_pitch_builds_record():
    log = PitchLog()
    record = log.add_pitch(10, 0.4, make_track(), 0, PitchCoordinate(x=63.0, y=41.5))
    assert len(log) == 1
    assert record.team == "TEAM_A"
    assert record.player_id == 7
    assert record.pitch_x == pytest.approx(63.0)
    assert record.pitch_y == pytest.approx(41.5)
    assert record.position == pytest.approx((63.0, 41.5))


def test_unknown_and_team_b_labels():
    log = PitchLog()
    log.add_pitch(1, 0.04, make_track(2), None, PitchCoordinate(1.0, 2.0))
    log.add_pitch(1, 0.04, make_track(3), 1, PitchCoordinate(3.0, 4.0))
    teams = {record.team for record in log}
    assert teams == {"UNKNOWN", "TEAM_B"}


def test_frame_and_id_queries():
    log = PitchLog()
    coord = PitchCoordinate(5.0, 5.0)
    log.add_pitch(0, 0.0, make_track(1), 0, coord)
    log.add_pitch(1, 0.04, make_track(1), 0, coord)
    log.add_pitch(1, 0.04, make_track(2), 1, coord)
    assert log.frames == {0, 1}
    assert log.unique_ids == {1, 2}
    assert len(log.for_frame(1)) == 2
    assert len(log.for_frame(0)) == 1
    assert log.to_dicts()[0]["pitch_x"] == 5.0


def test_save_csv_roundtrip(tmp_path: Path):
    log = PitchLog()
    log.add_pitch(100, 4.04, make_track(7), 0, PitchCoordinate(x=63.2, y=41.8))
    log.add_pitch(100, 4.04, make_track(9), None, PitchCoordinate(x=12.0, y=66.0))

    destination = log.save_csv(tmp_path / "nested" / "pitch.csv")
    assert destination.exists()

    with open(destination, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CSV_FIELDS
    assert len(rows) == 3  # header + 2 records
    assert rows[1][2] == "7"
    assert rows[1][3] == "TEAM_A"
    assert float(rows[1][4]) == pytest.approx(63.2)
    assert float(rows[1][5]) == pytest.approx(41.8)
