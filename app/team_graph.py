"""
Teammate Connection Graph (Stage 7)
===================================

Builds a *sparse, temporally stable* graph of connections between teammates on
the top-down pitch and computes network metrics. This module is pure logic -
it never draws anything; rendering lives in :mod:`app.pitch_renderer`.

Edge rule (per team, per frame)
--------------------------------
An undirected edge connects two players of the **same team** only when

1. their distance is at most ``max_connection_distance`` (metres), and
2. they are k-NN candidates: at least one of the two ranks the other among
   its ``k_neighbors`` closest teammates.

Together this keeps the graph local - no long edges across the pitch and no
fully-connected meshes - while still following the players as they move.
**Hysteresis**: once an edge exists it survives up to
``max_connection_distance * (1 + edge_hysteresis_ratio)`` before dropping, and
an active edge no longer needs the k-NN test. That band stops edges flickering
as players hover around the distance cap or swap neighbour ranks.

Metrics (per team)
------------------
``avg_teammate_distance`` / ``max_teammate_distance`` are all-pairs statistics
computed by :func:`app.formation_analyzer.teammate_distance_stats` - the exact
same helper Stage 6 uses, so the two stages can never disagree.
``connections`` (edge count) and ``density`` (edges / possible edges) come from
the graph itself.

No passing-lane detection is performed here: connections mean proximity-based
teammate structure, not ball movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from app.formation_analyzer import TEAM_LABELS, teammate_distance_stats

# (team, lower track id, higher track id)
EdgeKey = Tuple[str, int, int]


@dataclass(frozen=True)
class GraphEdge:
    """One teammate connection: ``team`` plus the two track ids it joins."""

    team: str
    id_a: int
    id_b: int
    distance: float


@dataclass
class TeamGraphResult:
    """Frame snapshot: the edges to draw plus per-team network metrics."""

    edges: List[GraphEdge] = field(default_factory=list)
    metrics: Dict[str, dict] = field(default_factory=dict)

    def pairs_for_team(self, team: str) -> List[Tuple[int, int]]:
        """Track-id pairs of one team, for renderers that draw by id."""
        return [(e.id_a, e.id_b) for e in self.edges if e.team == team]


class TeamGraphBuilder:
    """
    Temporally stable teammate graph builder.

    Call :meth:`update` once per frame; the builder keeps only the active-edge
    set between frames (both players must be present for an edge to exist).
    """

    def __init__(
        self,
        max_connection_distance: float = 25.0,
        k_neighbors: int = 3,
        hysteresis_ratio: float = 0.15,
    ) -> None:
        """
        Args:
            max_connection_distance: Metres beyond which teammates never link.
            k_neighbors: Each player links only to its k closest teammates
                (inside the distance cap) - keeps the graph sparse.
            hysteresis_ratio: Extra distance ratio an *established* edge may
                exceed the cap by before it is dropped (0.15 = +15%).
        """
        self.max_connection_distance = float(max_connection_distance)
        self.k_neighbors = max(1, int(k_neighbors))
        self.hysteresis_ratio = max(0.0, float(hysteresis_ratio))
        self._active: Set[EdgeKey] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def keep_threshold(self) -> float:
        """Distance an already-established edge may still reach."""
        return self.max_connection_distance * (1.0 + self.hysteresis_ratio)

    def update(self, positions: Sequence[tuple]) -> TeamGraphResult:
        """
        Build this frame's graph.

        Args:
            positions: ``(track_id, team_label, PitchCoordinate)`` tuples for
                every player with a pitch position this frame - the same
                input :meth:`app.formation_analyzer.FormationAnalyzer.update`
                consumes.

        Returns:
            :class:`TeamGraphResult` with deterministic edge order and one
            metrics entry per team.
        """
        grouped: Dict[str, Dict[int, Tuple[float, float]]] = {
            label: {} for label in TEAM_LABELS
        }
        for track_id, label, coordinate in positions:
            if label in grouped:
                grouped[label][track_id] = (coordinate.x, coordinate.y)

        edges: List[GraphEdge] = []
        metrics: Dict[str, dict] = {}
        next_active: Set[EdgeKey] = set()

        for label in TEAM_LABELS:
            nodes = grouped[label]
            distances, knn_pairs = self._neighbourhood(nodes)
            connections = 0

            for (id_a, id_b), distance in sorted(distances.items()):
                key = (label, id_a, id_b)
                established = key in self._active
                if established:
                    keep = distance <= self.keep_threshold
                else:
                    keep = (
                        distance <= self.max_connection_distance
                        and (id_a, id_b) in knn_pairs
                    )
                if keep:
                    next_active.add(key)
                    connections += 1
                    edges.append(
                        GraphEdge(team=label, id_a=id_a, id_b=id_b, distance=distance)
                    )

            metrics[label] = self._metrics(label, nodes, connections)

        self._active = next_active
        edges.sort(key=lambda e: (e.team, e.id_a, e.id_b))
        return TeamGraphResult(edges=edges, metrics=metrics)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _neighbourhood(
        self, nodes: Dict[int, Tuple[float, float]]
    ) -> Tuple[Dict[Tuple[int, int], float], Set[Tuple[int, int]]]:
        """
        Pairwise distances plus the k-nearest-neighbour edge candidates.

        Returns:
            (distances keyed by ``(lower id, higher id)``, candidate pairs).
        """
        ids = sorted(nodes)
        distances: Dict[Tuple[int, int], float] = {}
        neighbours: Dict[int, List[Tuple[float, int]]] = {i: [] for i in ids}

        for i, id_a in enumerate(ids):
            xa, ya = nodes[id_a]
            for id_b in ids[i + 1 :]:
                xb, yb = nodes[id_b]
                distance = math.hypot(xa - xb, ya - yb)
                key = (id_a, id_b)
                distances[key] = distance
                neighbours[id_a].append((distance, id_b))
                neighbours[id_b].append((distance, id_a))

        knn_pairs: Set[Tuple[int, int]] = set()
        for id_a in ids:
            for _distance, id_b in sorted(neighbours[id_a])[: self.k_neighbors]:
                knn_pairs.add((id_a, id_b) if id_a < id_b else (id_b, id_a))
        return distances, knn_pairs

    @staticmethod
    def _metrics(team: str, nodes: Dict[int, Tuple[float, float]], edges: int) -> dict:
        """Per-team network metrics (see module docstring)."""
        points = list(nodes.values())
        count = len(points)
        possible = count * (count - 1) // 2
        avg_distance, max_distance = teammate_distance_stats(points)
        return {
            "team": team,
            "players": count,
            "avg_teammate_distance": avg_distance,
            "max_teammate_distance": max_distance,
            "connections": edges,
            "density": (edges / possible) if possible else 0.0,
        }
