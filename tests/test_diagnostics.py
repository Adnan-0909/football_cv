"""
Unit Tests for Run Diagnostics
==============================

Covers the per-frame detection / tracking / team bookkeeping: track
lifecycles and gaps, label timelines (commits and A<->B switches per track),
and the end-of-run summary that attributes failures to detection vs tracking
vs team classification.

These are pure-numbers tests: diagnostics must never depend on video content
and must never influence the stages they observe.
"""

import json

import pytest

from app.diagnostics import RunDiagnostics, TrackRecord
from app.detector import Detection
from app.tracker import TrackedObject


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def det(class_name: str = "player", conf: float = 0.9) -> Detection:
    return Detection(
        bbox=(10.0, 10.0, 30.0, 60.0),
        confidence=conf,
        class_id=0 if class_name == "player" else 32,
        class_name=class_name,
    )


def trk(track_id: int, class_name: str = "player") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(10.0, 10.0, 30.0, 60.0),
        class_id=0 if class_name == "player" else 32,
        class_name=class_name,
        confidence=0.9,
    )


def record_frame(
    diag: RunDiagnostics,
    frame_index: int,
    ids=(),
    labels=None,
    n_det=None,
    extra_dets=0,
) -> None:
    """Feed one synthetic frame: `ids` tracks, `labels` override per ID."""
    labels = labels or {}
    detections = [det() for _ in range(n_det if n_det is not None else len(ids) + extra_dets)]
    tracks = [trk(i) for i in ids]
    diag.record(frame_index, detections, tracks, dict(labels))


# ---------------------------------------------------------------------- #
# Lifecycle
# ---------------------------------------------------------------------- #

def test_track_lifecycle_counts_frames_and_gaps():
    diag = RunDiagnostics()
    record_frame(diag, 0, ids=(1,))
    record_frame(diag, 1, ids=(1,))
    record_frame(diag, 2, ids=())       # detection gap
    record_frame(diag, 3, ids=(1,))     # same ID resumes

    record = diag.tracks[1]
    assert record.first_seen == 0
    assert record.last_seen == 3
    assert record.frames_active == 3
    assert record.max_gap == 1
    assert record.span == 4

    summary = diag.summary()["tracks"]
    assert summary["unique"] == 1
    assert summary["losses"] == 0       # last_seen == final frame
    assert summary["bridged_gaps"] == 1
    assert summary["max_gap"] == 1


def test_track_lost_before_the_final_frame_is_counted_as_a_loss():
    diag = RunDiagnostics()
    record_frame(diag, 0, ids=(1, 2))
    record_frame(diag, 1, ids=(2,))
    record_frame(diag, 2, ids=(2,))

    summary = diag.summary()["tracks"]
    assert summary["unique"] == 2
    assert summary["losses"] == 1       # track 1 vanished before frame 2


def test_detection_and_active_track_statistics():
    diag = RunDiagnostics()
    record_frame(diag, 0, ids=(1, 2), n_det=5)
    record_frame(diag, 1, ids=(), n_det=0)
    record_frame(diag, 2, ids=(3,), n_det=3)

    summary = diag.summary()
    assert summary["frames"] == 3
    assert summary["detections"]["mean"] == pytest.approx(8 / 3)
    assert summary["detections"]["max"] == 5
    assert summary["detections"]["zero_frames"] == 1
    assert summary["active_tracks"]["mean"] == pytest.approx(1.0)
    assert summary["active_tracks"]["max"] == 2


def test_balls_count_as_detections_but_never_as_tracks():
    diag = RunDiagnostics()
    diag.record(0, [det(), det("ball")], [trk(1), trk(2, "ball")], {})

    assert 1 in diag.tracks and 2 not in diag.tracks
    summary = diag.summary()
    assert summary["detections"]["mean"] == 2
    assert summary["detections"]["player_mean"] == pytest.approx(1.0)
    assert summary["active_tracks"]["max"] == 1


def test_display_grade_band_is_recorded_separately():
    """Candidates and display-grade detections stay separable, so runs with a
    lowered detector floor remain comparable to display-only runs."""
    diag = RunDiagnostics(display_threshold=0.35)
    diag.record(0, [det(conf=0.2), det(conf=0.9)], [], {})
    summary = diag.summary()["detections"]
    assert summary["mean"] == 2                       # both are candidates
    assert summary["display_threshold"] == pytest.approx(0.35)
    assert summary["display_mean"] == pytest.approx(1.0)
    assert summary["display_zero_frames"] == 0

    diag.record(1, [det(conf=0.2)], [], {})
    summary = diag.summary()["detections"]
    assert summary["zero_frames"] == 0                # frame is not "empty"
    assert summary["display_zero_frames"] == 1        # ...but has no display-grade hit
    assert summary["display_mean"] == pytest.approx(0.5)
    assert "display-grade" in diag.text()


# ---------------------------------------------------------------------- #
# Team-label timelines
# ---------------------------------------------------------------------- #

def test_commits_and_team_switches_are_counted_per_track():
    diag = RunDiagnostics()
    record_frame(diag, 0, ids=(1,), labels={1: None})
    record_frame(diag, 1, ids=(1,), labels={1: None})   # unchanged -> no event
    record_frame(diag, 2, ids=(1,), labels={1: 0})      # commit UNKNOWN -> A
    record_frame(diag, 3, ids=(1,), labels={1: 0})
    record_frame(diag, 4, ids=(1,), labels={1: 1})      # switch A -> B

    record = diag.tracks[1]
    assert record.label_changes == [(0, None), (2, 0), (4, 1)]
    assert record.label_events() == (1, 1, 0)
    assert record.final_label == 1

    teams = diag.summary()["teams"]
    assert teams["commits_total"] == 1
    assert teams["ab_switches_total"] == 1
    assert teams["withdrawals_total"] == 0
    assert teams["tracks_with_switch"] == 1
    assert teams["switches_per_track"] == {1: 1}
    # Switched once -> not counted as a stable Team B track.
    assert teams["stable_TEAM_B"] == 0


def test_stable_and_unknown_final_labels():
    diag = RunDiagnostics()
    for frame in range(3):
        record_frame(diag, frame, ids=(1, 2), labels={1: 0, 2: None})
    record_frame(diag, 3, ids=(1, 2), labels={1: 1, 2: None})  # track 1 flips late

    teams = diag.summary()["teams"]
    assert teams["final_TEAM_A"] == 0
    assert teams["final_TEAM_B"] == 1
    assert teams["final_UNKNOWN"] == 1
    assert teams["stable_TEAM_B"] == 0   # flip-flopped
    assert diag.tracks[1].label_events() == (0, 1, 0)


def test_track_that_never_flips_counts_as_stable():
    diag = RunDiagnostics()
    for frame in range(4):
        record_frame(diag, frame, ids=(1, 2), labels={1: 1, 2: 0})

    teams = diag.summary()["teams"]
    assert teams["stable_TEAM_A"] == 1
    assert teams["stable_TEAM_B"] == 1
    assert teams["ab_switches_total"] == 0


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #

def test_summary_is_json_serializable_and_text_report_has_sections():
    diag = RunDiagnostics()
    record_frame(diag, 0, ids=(1,), labels={1: 0})
    record_frame(diag, 1, ids=(1,), labels={1: 1})

    payload = json.dumps(diag.summary())
    assert "switches_per_track" in payload

    text = diag.text()
    for section in (
        "frames processed",
        "detections / frame",
        "tracks",
        "team-label events",
        "switches per track",
    ):
        assert section in text


def test_empty_run_reports_zeros_without_crashing():
    diag = RunDiagnostics()
    summary = diag.summary()
    assert summary["frames"] == 0
    assert summary["tracks"]["unique"] == 0
    assert summary["teams"]["ab_switches_total"] == 0
    assert "frames processed" in diag.text()
