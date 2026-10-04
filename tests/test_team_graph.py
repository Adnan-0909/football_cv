"""
Stage 7: Teammate Connection Graph Tests
========================================

Covers the k-NN + distance-cap graph, same-team-only edges, temporal
hysteresis (no per-frame flicker), and the network metrics.
"""

import pytest

from app.formation_analyzer import FormationAnalyzer, teammate_distance_stats
from app.pitch import PitchCoordinate
from app.team_graph import GraphEdge, TeamGraphBuilder, TeamGraphResult

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"


def positions(entries, team=TEAM_A, start_id=1):
    """``[(x, y), ...]`` -> ``(track_id, team_label, PitchCoordinate)`` tuples."""
    return [
        (start_id + i, team, PitchCoordinate(x=float(x), y=float(y)))
        for i, (x, y) in enumerate(entries)
    ]


def edge_pairs(result: TeamGraphResult, team=TEAM_A):
    return {(e.id_a, e.id_b) for e in result.edges if e.team == team}


# ---------------------------------------------------------------------- #
# Edge rules
# ---------------------------------------------------------------------- #

def test_edges_connect_same_team_players_only():
    # Two teams, two close teammates each - never a cross-team edge.
    frame = positions([(20, 20), (25, 20)], TEAM_A, start_id=1) + positions(
        [(60, 40), (65, 40)], TEAM_B, start_id=3
    )
    result = TeamGraphBuilder().update(frame)

    assert edge_pairs(result, TEAM_A) == {(1, 2)}
    assert edge_pairs(result, TEAM_B) == {(3, 4)}
    ids_a = {1, 2}
    ids_b = {3, 4}
    for edge in result.edges:
        members = ids_a if edge.team == TEAM_A else ids_b
        assert edge.id_a in members and edge.id_b in members


def test_no_edges_beyond_max_connection_distance():
    # 30 m apart with a 25 m cap: k-NN proposes, the distance filter vetoes.
    frame = positions([(10, 34), (40, 34), (70, 34)])
    result = TeamGraphBuilder(max_connection_distance=25.0).update(frame)

    assert result.edges == []
    assert result.metrics[TEAM_A]["connections"] == 0
    assert result.metrics[TEAM_A]["density"] == 0.0


def test_max_connection_distance_is_configurable():
    frame = positions([(10, 34), (30, 34)])  # 20 m apart

    strict = TeamGraphBuilder(max_connection_distance=15.0).update(frame)
    loose = TeamGraphBuilder(max_connection_distance=25.0).update(frame)

    assert strict.edges == []
    assert edge_pairs(loose, TEAM_A) == {(1, 2)}


def test_knn_keeps_dense_cluster_from_becoming_fully_connected():
    """10 players all inside the cap -> links, but the k-NN keeps it sparse."""
    frame = positions([(20 + 2 * i, 20 + 2 * (i % 4)) for i in range(10)])
    result = TeamGraphBuilder(max_connection_distance=25.0, k_neighbors=3).update(frame)

    connections = result.metrics[TEAM_A]["connections"]
    assert connections > 0
    # Every one of the 45 possible pairs is within the cap - only the k-NN
    # relation stops the graph from filling in (3 directed proposals per
    # player, so at most 30 undirected edges).
    assert 0 < connections <= 30 < 45
    assert result.metrics[TEAM_A]["density"] < 1.0


def test_isolated_player_gets_no_edges():
    tight = positions([(20, 20), (24, 20), (22, 24)], TEAM_A, start_id=1)
    loner = positions([(95, 60)], TEAM_A, start_id=10)

    result = TeamGraphBuilder().update(tight + loner)

    assert 10 not in {i for pair in edge_pairs(result) for i in pair}
    assert result.metrics[TEAM_A]["players"] == 4


# ---------------------------------------------------------------------- #
# Temporal stability (hysteresis)
# ---------------------------------------------------------------------- #

def test_established_edge_survives_the_release_band():
    """Once formed at <= 25 m, an edge survives out to 25 * 1.15 = 28.75 m."""
    builder = TeamGraphBuilder(max_connection_distance=25.0, hysteresis_ratio=0.15)

    assert edge_pairs(builder.update(positions([(10, 34), (34, 34)]))) == {(1, 2)}  # 24 m

    # 27 m: a brand-new edge would be rejected, the existing one persists.
    builder.update(positions([(10, 34), (37, 34)]))
    assert edge_pairs(builder.update(positions([(10, 34), (37, 34)]))) == {(1, 2)}

    # Past the band: dropped.
    assert builder.update(positions([(10, 34), (41, 34)])).edges == []  # 31 m


def test_new_edge_does_not_get_the_release_band():
    builder = TeamGraphBuilder(max_connection_distance=25.0, hysteresis_ratio=0.15)

    # 27 m on the very first observation: never established, so no edge.
    assert builder.update(positions([(10, 34), (37, 34)])).edges == []


def test_edge_reappears_when_both_players_return():
    builder = TeamGraphBuilder()
    two = positions([(20, 20), (24, 20)], TEAM_A, start_id=1)
    only_one = positions([(20, 20)], TEAM_A, start_id=1)

    assert edge_pairs(builder.update(two)) == {(1, 2)}
    assert builder.update(only_one).edges == []  # an endpoint vanished
    assert edge_pairs(builder.update(two)) == {(1, 2)}  # both back -> edge back


def test_results_are_deterministically_ordered():
    builder = TeamGraphBuilder()
    frame = positions([(20, 20), (24, 20), (24, 40)], TEAM_A) + positions(
        [(60, 20), (64, 20), (64, 40)], TEAM_B, start_id=3
    )

    first = builder.update(frame)
    second = TeamGraphBuilder().update(frame)

    keys = [(e.team, e.id_a, e.id_b) for e in first.edges]
    assert keys == sorted(keys)
    assert keys == [(e.team, e.id_a, e.id_b) for e in second.edges]


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

def test_metrics_report_connections_density_and_distances():
    frame = positions([(20, 20), (25, 20), (20, 25), (90, 60)])
    result = TeamGraphBuilder(max_connection_distance=25.0, k_neighbors=3).update(frame)
    metric = result.metrics[TEAM_A]

    assert metric["team"] == TEAM_A
    assert metric["players"] == 4
    assert metric["connections"] == 3  # the tight triangle only
    possible = 4 * 3 // 2
    assert metric["density"] == pytest.approx(3 / possible)

    points = [(20.0, 20.0), (25.0, 20.0), (20.0, 25.0), (90.0, 60.0)]
    avg, biggest = teammate_distance_stats(points)
    assert metric["avg_teammate_distance"] == pytest.approx(avg)
    assert metric["max_teammate_distance"] == pytest.approx(biggest)


def test_avg_teammate_distance_matches_stage6():
    """Stage 6 and Stage 7 must never disagree on the same frame."""
    frame = positions([(20, 10), (21, 25), (22, 43), (23, 58), (50, 34)])

    stage6 = FormationAnalyzer().update(frame)[TEAM_A]["avg_teammate_distance"]
    stage7 = TeamGraphBuilder().update(frame).metrics[TEAM_A]["avg_teammate_distance"]

    assert stage7 == pytest.approx(stage6)


def test_metrics_for_team_without_players():
    result = TeamGraphBuilder().update(positions([(20, 20)], TEAM_B))
    metric = result.metrics[TEAM_A]

    assert metric["players"] == 0
    assert metric["connections"] == 0
    assert metric["density"] == 0.0
    assert metric["avg_teammate_distance"] == 0.0
    assert metric["max_teammate_distance"] == 0.0


def test_result_object_helpers():
    result = TeamGraphBuilder().update(
        positions([(20, 20), (24, 20)], TEAM_A)
    )
    assert isinstance(result, TeamGraphResult)
    assert result.pairs_for_team(TEAM_A) == [(1, 2)]
    assert result.pairs_for_team(TEAM_B) == []
    assert isinstance(result.edges[0], GraphEdge)
    assert result.edges[0].distance == pytest.approx(4.0)
