"""
Unit Tests for the ByteTrack-lite Player Tracker
================================================

Synthetic detections (no video, no model) exercise ID stability, occlusion
recovery, threshold behaviour, and class separation.
"""

import pytest

from app.config import TrackerConfig
from app.detector import Detection
from app.tracker import PlayerTracker


def make_det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 0.9,
    class_id: int = 0,
) -> Detection:
    class_name = "player" if class_id == 0 else "ball"
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=float(conf),
        class_id=class_id,
        class_name=class_name,
    )


def test_first_frame_creates_a_track():
    tracker = PlayerTracker()
    tracks = tracker.update([make_det(10, 10, 50, 100)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "player"
    assert tracks[0].trajectory == [(30.0, 55.0)]


def test_ids_stay_stable_while_object_moves():
    tracker = PlayerTracker()
    seen_ids = []
    for step in range(20):
        x = 10 + step * 5
        tracks = tracker.update([make_det(x, 10, x + 40, 100)])
        assert len(tracks) == 1, f"frame {step}: expected exactly one active track"
        seen_ids.append(tracks[0].track_id)
    assert len(set(seen_ids)) == 1


def test_id_survives_a_short_occlusion():
    tracker = PlayerTracker()
    tracker.update([make_det(10, 10, 50, 100)])
    for _ in range(5):
        tracks = tracker.update([])  # fully occluded
        assert tracks == []

    # Reappears 25px away: still the same object.
    tracks = tracker.update([make_det(25, 10, 65, 100)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_track_is_discarded_after_the_buffer_window():
    # Past the buffer (+2: the upstream pool evicts a removed track one frame
    # later than it is marked), so the stale track is really gone.
    config = TrackerConfig()
    tracker = PlayerTracker(config)
    tracker.update([make_det(10, 10, 50, 100)])
    for _ in range(config.track_buffer + 2):
        tracker.update([])

    # ByteTrack confirms a brand-new track on its second hit, so allow two
    # frames for the replacement ID to be reported.
    seen = []
    for _ in range(2):
        seen.extend(t.track_id for t in tracker.update([make_det(10, 10, 50, 100)]))
    assert set(seen) == {2}, "stale track should have been removed, forcing a new ID"


def test_low_confidence_detection_reuses_the_existing_track():
    tracker = PlayerTracker()
    tracker.update([make_det(10, 10, 50, 100, conf=0.9)])

    # 0.35 sits in the low-confidence band: stage 2 must rescue it without
    # spawning a second track.
    tracks = tracker.update([make_det(10, 10, 50, 100, conf=0.35)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert len(tracker.live_tracks) == 1


def test_low_confidence_detection_does_not_create_tracks():
    tracker = PlayerTracker()
    tracks = tracker.update([make_det(10, 10, 50, 100, conf=0.4)])
    assert tracks == []
    assert tracker.live_tracks == []


def test_new_track_requires_new_track_thresh():
    config = TrackerConfig()
    tracker = PlayerTracker(config)

    # High-confidence band, but below new_track_thresh -> ignored.
    tracks = tracker.update([make_det(10, 10, 50, 100, conf=config.new_track_thresh - 0.01)])
    assert tracks == []
    assert tracker.live_tracks == []

    # Crossing the threshold starts a track (reported once confirmed, i.e. on
    # the second hit for backends that filter unconfirmed tracks).
    seen = []
    for _ in range(2):
        seen.extend(t.track_id for t in tracker.update([make_det(10, 10, 50, 100, conf=0.9)]))
    assert set(seen) == {1}


def test_players_and_balls_keep_separate_ids():
    tracker = PlayerTracker()
    tracker.update(
        [
            make_det(10, 10, 50, 100, class_id=0),
            make_det(10, 10, 50, 100, class_id=32),
        ]
    )
    tracks = tracker.update(
        [
            make_det(10, 10, 50, 100, class_id=0),
            make_det(10, 10, 50, 100, class_id=32),
        ]
    )
    players = [t for t in tracks if t.class_name == "player"]
    balls = [t for t in tracks if t.class_name == "ball"]
    assert len(players) == 1 and len(balls) == 1
    assert players[0].track_id != balls[0].track_id


def test_trajectory_is_capped_at_track_buffer():
    config = TrackerConfig(track_buffer=10)
    tracker = PlayerTracker(config)
    for step in range(40):
        x = 10 + step * 3
        tracker.update([make_det(x, 10, x + 40, 100)])
    assert len(tracker.live_tracks[0].trajectory) <= 10


def test_reset_restarts_ids():
    tracker = PlayerTracker()
    tracker.update([make_det(10, 10, 50, 100)])
    tracker.reset()
    assert tracker.live_tracks == []
    assert tracker.frame_index == 0
    tracks = tracker.update([make_det(10, 10, 50, 100)])
    assert tracks[0].track_id == 1


def test_empty_frame_without_prior_tracks_is_safe():
    tracker = PlayerTracker()
    assert tracker.update([]) == []


# ---------------------------------------------------------------------- #
# Backend selection & Stage 2 requirements
# ---------------------------------------------------------------------- #

def test_default_backend_is_the_ultralytics_bytetrack():
    """The built-in, maintained ByteTrack is what users get by default."""
    from app.tracker import _UltralyticsByteTrack

    tracker = PlayerTracker()
    assert tracker.config.tracker_type == "bytetrack"
    assert isinstance(tracker._impl, _UltralyticsByteTrack)
    assert tracker.backend == "bytetrack"


def test_lite_backend_is_selectable():
    from app.tracker import _ByteTrackLite

    tracker = PlayerTracker(TrackerConfig(tracker_type="bytetrack_lite"))
    assert isinstance(tracker._impl, _ByteTrackLite)
    tracker.update([make_det(10, 10, 50, 100)])
    assert [t.track_id for t in tracker.live_tracks] == [1]


def test_unknown_tracker_type_raises():
    with pytest.raises(ValueError, match="tracker_type"):
        PlayerTracker(TrackerConfig(tracker_type="botsort"))


def test_tracked_object_carries_detection_confidence():
    tracker = PlayerTracker()
    tracks = tracker.update([make_det(10, 10, 50, 100, conf=0.9)])
    assert len(tracks) == 1
    assert tracks[0].confidence == pytest.approx(0.9, abs=1e-3)


@pytest.mark.parametrize("tracker_type", ["bytetrack", "bytetrack_lite"])
def test_players_moving_in_parallel_keep_distinct_stable_ids(tracker_type):
    """Four players drifting across the pitch never swap IDs (Stage 2 core goal)."""
    tracker = PlayerTracker(TrackerConfig(tracker_type=tracker_type))
    starts = [100.0, 400.0, 700.0, 950.0]
    velocities = [3.0, -2.0, 1.5, -4.0]

    history: list[dict[int, tuple]] = []
    for step in range(30):
        detections = [
            make_det(start + vel * step, 200, start + vel * step + 40, 290)
            for start, vel in zip(starts, velocities)
        ]
        tracks = tracker.update(detections)
        assert len(tracks) == 4, f"frame {step}: expected 4 active tracks"
        history.append({t.track_id: t.bbox for t in tracks})

    assert len(history[0]) == 4
    # Every ID seen in the first frame must be the only ID ever seen for that
    # player - i.e. no ID was ever handed to a second player.
    first_frame_ids = set(history[0])
    all_ids = set().union(*(set(frame) for frame in history))
    assert all_ids == first_frame_ids, f"IDs churned during the run: {sorted(all_ids)}"

    # Each ID must stay attached to the same physical player (its box must keep
    # moving smoothly rather than jumping to another player's position).
    for track_id in first_frame_ids:
        previous = history[0][track_id]
        for frame in history[1:]:
            box = frame[track_id]
            displacement = max(abs(box[i] - previous[i]) for i in range(4))
            assert displacement <= 20, f"ID {track_id} jumped {displacement:.0f}px - ID switch"
            previous = box


@pytest.mark.parametrize("tracker_type", ["bytetrack", "bytetrack_lite"])
def test_id_survives_occlusion_on_both_backends(tracker_type):
    tracker = PlayerTracker(TrackerConfig(tracker_type=tracker_type))
    tracker.update([make_det(10, 10, 50, 100)])
    for _ in range(8):
        assert tracker.update([]) == []

    tracks = tracker.update([make_det(30, 10, 70, 100)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
