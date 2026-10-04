"""
Stage 6: Formation Analysis Tests
=================================

Covers the sliding-window formation recognizer: direction detection, line
splitting, template matching, confidence/UNKNOWN policy, temporal smoothing,
and the shared teammate-distance helper.
"""

import math

import pytest

from app.formation_analyzer import (
    FORMATION_TEMPLATES,
    FormationAnalyzer,
    teammate_distance_stats,
)
from app.pitch import PitchCoordinate

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"


def positions(entries, team=TEAM_A, start_id=1):
    """``[(x, y), ...]`` -> ``(track_id, team_label, PitchCoordinate)`` tuples."""
    return [
        (start_id + i, team, PitchCoordinate(x=float(x), y=float(y)))
        for i, (x, y) in enumerate(entries)
    ]


def block_433(attack_right=True):
    """11 players in a clean 4-3-3 shape (GK + three tight lines)."""
    defenders = [(20, 10), (21, 25), (22, 43), (23, 58)]
    midfielders = [(50, 20), (51, 34), (52, 48)]
    attackers = [(78, 16), (79, 34), (80, 52)]
    outfield = defenders + midfielders + attackers
    if not attack_right:
        outfield = [(105 - x, y) for x, y in outfield]
        outfield = sorted(outfield)  # keep it human-readable, order is irrelevant
    keeper = (5, 34) if attack_right else (100, 34)
    return [keeper] + outfield


def block_442():
    """11 players in a clean 4-4-2 shape."""
    outfield = (
        [(20, 10), (21, 25), (22, 43), (23, 58)]       # 4 defenders
        + [(50, 12), (51, 28), (52, 42), (53, 57)]     # 4 midfielders
        + [(78, 24), (80, 44)]                          # 2 strikers
    )
    return [(5, 34)] + outfield


# ---------------------------------------------------------------------- #
# Shared helper
# ---------------------------------------------------------------------- #

def test_teammate_distance_stats_known_triangle():
    avg, biggest = teammate_distance_stats([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)])
    assert avg == pytest.approx((3 + 4 + 5) / 3)
    assert biggest == pytest.approx(5.0)


def test_teammate_distance_stats_single_and_empty():
    assert teammate_distance_stats([]) == (0.0, 0.0)
    assert teammate_distance_stats([(10.0, 20.0)]) == (0.0, 0.0)


# ---------------------------------------------------------------------- #
# Result shape / defaults
# ---------------------------------------------------------------------- #

def test_fresh_analyzer_reports_unknown_for_both_teams():
    results = FormationAnalyzer().update([])

    assert set(results) == {TEAM_A, TEAM_B}
    required = {
        "team", "formation", "confidence", "width", "depth",
        "compactness", "avg_teammate_distance", "players",
    }
    for label in (TEAM_A, TEAM_B):
        assert set(results[label]) == required
        assert results[label]["formation"] == "UNKNOWN"
        assert results[label]["confidence"] == 0.0
        assert results[label]["players"] == 0


def test_templates_are_all_three_tuples_summing_to_ten():
    for name, template in FORMATION_TEMPLATES.items():
        assert len(template) == 3, name
        assert sum(template) == 10, name


# ---------------------------------------------------------------------- #
# Template recognition
# ---------------------------------------------------------------------- #

def test_detects_433_when_attacking_right():
    analyzer = FormationAnalyzer()
    results = analyzer.update(positions(block_433()) + positions(block_433(), TEAM_B, start_id=50))

    team_a = results[TEAM_A]
    assert team_a["formation"] == "4-3-3"
    assert team_a["confidence"] >= 0.6
    assert team_a["players"] == 11  # goalkeeper included in the count

    # Metrics cover every visible player (GK at x=5 counts for depth):
    # width across the pitch, depth from the deepest to the most advanced.
    assert team_a["width"] == pytest.approx(58 - 10)
    assert team_a["depth"] == pytest.approx(80 - 5)
    assert team_a["avg_teammate_distance"] > 0
    assert team_a["compactness"] == pytest.approx(1.0 / (team_a["avg_teammate_distance"] + 0.1))


def test_detects_433_when_attacking_left():
    """The same shape mirrored must still be read as 4-3-3 (direction flips)."""
    results = FormationAnalyzer().update(positions(block_433(attack_right=False)))

    assert results[TEAM_A]["formation"] == "4-3-3"
    assert results[TEAM_A]["confidence"] >= 0.6


def test_detects_442():
    results = FormationAnalyzer().update(positions(block_442()))

    assert results[TEAM_A]["formation"] == "4-4-2"
    assert results[TEAM_A]["confidence"] >= 0.6


def test_unknown_when_too_few_players():
    # 3 outfield players cannot form three meaningful lines.
    results = FormationAnalyzer().update(positions([(20, 10), (50, 34), (80, 58)]))

    assert results[TEAM_A]["formation"] == "UNKNOWN"
    assert results[TEAM_A]["players"] == 3


def test_unknown_when_players_form_one_uniform_blob():
    """Evenly-spread players with no line structure must not be guessed."""
    blob = [(5, 34)] + [(x, 10 + (x % 50)) for x in range(15, 100, 8)]
    results = FormationAnalyzer().update(positions(blob))

    assert results[TEAM_A]["formation"] == "UNKNOWN"


def test_min_confidence_is_configurable():
    strict = FormationAnalyzer(min_confidence=0.99)
    lenient = FormationAnalyzer(min_confidence=0.5)

    assert strict.update(positions(block_433()))[TEAM_A]["formation"] == "UNKNOWN"
    assert lenient.update(positions(block_433()))[TEAM_A]["formation"] == "4-3-3"


# ---------------------------------------------------------------------- #
# Temporal smoothing
# ---------------------------------------------------------------------- #

def test_single_odd_frame_does_not_flip_the_label():
    """8 x 4-3-3 then one 4-4-2 -> the 4-3-3 label survives the vote."""
    analyzer = FormationAnalyzer(window_size=10)
    results = None
    for _ in range(8):
        results = analyzer.update(positions(block_433()))
    results = analyzer.update(positions(block_442()))

    assert results[TEAM_A]["formation"] == "4-3-3"
    # agreement (8/9) scaled into the reported confidence
    assert results[TEAM_A]["confidence"] < 1.0


def test_metrics_are_averaged_over_the_window():
    analyzer = FormationAnalyzer(window_size=4)
    analyzer.update(positions(block_433()))
    results = analyzer.update(positions(block_442()))

    # Both blocks run from the goalkeeper (x=5) to a striker at x=80, so the
    # window mean depth stays 75 m either way.
    assert results[TEAM_A]["depth"] == pytest.approx(75.0)


def test_window_size_of_one_follows_the_current_frame():
    analyzer = FormationAnalyzer(window_size=1)

    assert analyzer.update(positions(block_433()))[TEAM_A]["formation"] == "4-3-3"
    assert analyzer.update(positions(block_442()))[TEAM_A]["formation"] == "4-4-2"


def test_team_absent_frames_keep_previous_state():
    analyzer = FormationAnalyzer(window_size=5)
    analyzer.update(positions(block_433()))

    results = analyzer.update([])  # team vanished for a frame

    assert results[TEAM_A]["formation"] == "4-3-3"
    assert results[TEAM_A]["players"] == 11  # last known count


def test_unknown_labels_are_ignored():
    """Players whose team is not yet committed never feed the analyzer."""
    results = FormationAnalyzer().update(positions(block_433(), team="UNKNOWN"))

    assert results[TEAM_A]["players"] == 0
    assert results[TEAM_B]["players"] == 0
