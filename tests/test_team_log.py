"""
Unit Tests for Per-Frame Team Records (Stage 3)
================================================

Verifies the ``TeamLog`` export contract: exact CSV columns
``frame,timestamp,player_id,team,cx,cy``, the TEAM_A / TEAM_B / UNKNOWN label
vocabulary and a lossless round-trip.
"""

import csv
from pathlib import Path

import pytest

from app.team_classifier import TEAM_A, TEAM_B, team_label
from app.team_log import CSV_FIELDS, TeamLog, TeamRecord
from app.tracker import TrackedObject


def make_track(track_id: int = 7) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(10.0, 20.0, 30.0, 80.0),
        class_id=0,
        class_name="player",
        confidence=0.9,
    )


def test_csv_columns_match_the_documented_contract():
    assert list(CSV_FIELDS) == ["frame", "timestamp", "player_id", "team", "cx", "cy"]


def test_record_from_track_maps_raw_team_ids_to_labels():
    record = TeamRecord.from_track(frame=3, timestamp=0.12, track=make_track(7), team=TEAM_A)
    assert record.team == "TEAM_A"
    assert record.center == (20.0, 50.0)

    unknown = TeamRecord.from_track(frame=3, timestamp=0.12, track=make_track(8), team=None)
    assert unknown.team == "UNKNOWN"

    other = TeamRecord.from_track(frame=3, timestamp=0.12, track=make_track(9), team=TEAM_B)
    assert other.team == "TEAM_B"


def test_record_rejects_unknown_labels():
    with pytest.raises(ValueError):
        TeamRecord(frame=0, timestamp=0.0, player_id=1, team="RED", center=(1.0, 2.0))


def test_record_schema_round_trip():
    record = TeamRecord(frame=5, timestamp=0.2, player_id=3, team="TEAM_A", center=(4.5, 6.5))
    data = record.to_dict()
    assert list(data) == ["frame", "timestamp", "player_id", "team", "center"]
    assert data["team"] == "TEAM_A"
    assert record.to_row() == (5, 0.2, 3, "TEAM_A", 4.5, 6.5)


def test_log_collects_records_and_counts(tmp_path: Path):
    log = TeamLog()
    log.add_team(0, 0.0, make_track(1), TEAM_A)
    log.add_team(0, 0.0, make_track(2), TEAM_B)
    log.add_team(1, 0.04, make_track(1), TEAM_A)
    log.add_team(1, 0.04, make_track(3), None)

    assert len(log) == 4
    assert log.unique_ids == {1, 2, 3}
    assert log.frames == {0, 1}
    assert len(log.for_frame(1)) == 2

    # Unique players per label across the run.
    assert log.team_counts == {"TEAM_A": 1, "TEAM_B": 1, "UNKNOWN": 1}


def test_save_csv_writes_exact_columns_and_round_trips(tmp_path: Path):
    log = TeamLog()
    log.add_team(0, 0.0, make_track(1), TEAM_A)
    log.add_team(1, 0.04, make_track(1), None)

    path = log.save_csv(tmp_path / "nested" / "teams.csv")
    assert path.exists()

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["frame", "timestamp", "player_id", "team", "cx", "cy"]
    assert len(rows) == 3
    assert rows[1] == ["0", "0.0", "1", "TEAM_A", "20.0", "50.0"]
    assert rows[2][3] == "UNKNOWN"

    reloaded = TeamLog.from_csv(path)
    assert len(reloaded) == 2
    assert reloaded.records[0].center == (20.0, 50.0)
    assert reloaded.records[1].team == "UNKNOWN"


def test_empty_log_still_writes_the_header(tmp_path: Path):
    path = TeamLog().save_csv(tmp_path / "empty.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [list(CSV_FIELDS)]


def test_labels_come_from_the_shared_vocabulary():
    assert {team_label(TEAM_A), team_label(TEAM_B), team_label(None)} == {
        "TEAM_A",
        "TEAM_B",
        "UNKNOWN",
    }
