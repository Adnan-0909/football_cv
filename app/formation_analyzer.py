"""
Formation Analysis (Stage 6)
============================

Turns per-frame top-down pitch positions into a *temporally smoothed* formation
label per team::

    {
        "team": "TEAM_A",
        "formation": "4-3-3",   # or "UNKNOWN" when confidence is low
        "confidence": 0.87,
        "width": 41.2,          # metres across the pitch
        "depth": 33.5,          # metres along the pitch (own goal -> attack)
        "compactness": 0.09,    # 1 / (average teammate distance + 0.1)
        "avg_teammate_distance": 11.4,
        "players": 10,
    }

Shape metrics (width, depth, compactness, average teammate distance) cover
*every* visible team player - including the goalkeeper - which is exactly the
set Stage 7 uses, so both stages always report the same numbers. The keeper
is excluded only when splitting the team into three lines.

Method (pure Python/NumPy - no OpenCV, no scipy)
------------------------------------------------
1. **Direction**: a team's own goal is the goal line its extreme player sits
   closest to (normally the goalkeeper is the deepest player). Every outfield
   player is then expressed as *progress* = distance from the own goal line
   along the attacking direction.
2. **Lines**: players bunch into defensive / middle / attacking bands, so the
   two largest gaps in the sorted progress list split the team into three
   lines. This adapts to the actual shape instead of fixed pitch zones.
3. **Template**: the three line counts are matched against standard
   formations (all 3-tuples: defenders, midfielders, attackers - the
   goalkeeper is excluded). Confidence combines match accuracy with how
   cleanly the bands separate; below ``min_confidence`` the label is
   ``UNKNOWN`` rather than a guess.
4. **Smoothing**: raw per-frame labels go into a sliding window; the reported
   formation is the window's majority vote scaled by its agreement, so one
   odd frame never flips the overlay.

``teammate_distance_stats`` is shared with Stage 7 (:mod:`app.team_graph`) so
both stages report the *same* average/maximum teammate distance.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Deque, Dict, List, Sequence, Tuple

from app.pitch import PitchCoordinate
from app.team_classifier import TEAM_A, TEAM_B, team_label

# String labels used as keys in results ("TEAM_A" / "TEAM_B").
TEAM_LABELS: Tuple[str, ...] = (team_label(TEAM_A), team_label(TEAM_B))

# Standard formations as 3-tuples: (defenders, midfielders, attackers).
# The goalkeeper is never counted, so every template sums to 10 outfielders.
FORMATION_TEMPLATES: Dict[str, Tuple[int, int, int]] = {
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "5-3-2": (5, 3, 2),
    "5-4-1": (5, 4, 1),
    "4-5-1": (4, 5, 1),
}

# Below this many outfield players a three-line formation is not meaningful.
MIN_OUTFIELD_PLAYERS = 3

# compactness = 1 / (average teammate distance + FLOOR)
_COMPACTNESS_FLOOR = 0.1


def teammate_distance_stats(
    points: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Average and maximum pairwise Euclidean distance of a set of positions.

    Shared by Stage 6 (compactness) and Stage 7 (network metrics) so the two
    stages can never report inconsistent "average teammate distance" values.

    Args:
        points: (x, y) positions in metres.

    Returns:
        (average, maximum) distance in metres; ``(0.0, 0.0)`` for fewer than
        two points.
    """
    n = len(points)
    if n < 2:
        return 0.0, 0.0
    total = 0.0
    biggest = 0.0
    count = 0
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            total += d
            count += 1
            if d > biggest:
                biggest = d
    return total / count, biggest


def _empty_result(team: str) -> dict:
    """Result used before a team has been observed at all."""
    return {
        "team": team,
        "formation": "UNKNOWN",
        "confidence": 0.0,
        "width": 0.0,
        "depth": 0.0,
        "compactness": 0.0,
        "avg_teammate_distance": 0.0,
        "players": 0,
    }


class FormationAnalyzer:
    """
    Sliding-window formation recognition for both teams.

    Feed one frame per call to :meth:`update`; the returned dictionaries are
    ready for display (overlay) or logging. The analyzer keeps its own window,
    so no external state has to be threaded through the pipeline.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        window_size: int = 10,
        min_confidence: float = 0.6,
    ) -> None:
        """
        Args:
            pitch_length: Pitch length in metres (x extent).
            pitch_width: Pitch width in metres (y extent).
            window_size: Frames kept per team for temporal smoothing.
            min_confidence: Smoothed confidence floor for a non-UNKNOWN label.
        """
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.window_size = max(1, int(window_size))
        self.min_confidence = float(min_confidence)
        self._window: Dict[str, Deque[dict]] = {
            label: deque(maxlen=self.window_size) for label in TEAM_LABELS
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, positions: Sequence[tuple]) -> Dict[str, dict]:
        """
        Analyse one frame of player positions.

        Args:
            positions: ``(track_id, team_label, PitchCoordinate)`` tuples for
                every player that has a pitch position this frame. Entries
                whose team label is not ``TEAM_A``/``TEAM_B`` are ignored.

        Returns:
            ``{"TEAM_A": {...}, "TEAM_B": {...}}`` - each value has the keys
            documented in the module docstring.
        """
        grouped: Dict[str, List[PitchCoordinate]] = {label: [] for label in TEAM_LABELS}
        for _track_id, label, coordinate in positions:
            if label in grouped:
                grouped[label].append(coordinate)

        results: Dict[str, dict] = {}
        for label in TEAM_LABELS:
            players = grouped[label]
            if players:
                self._window[label].append(self._analyse_team(label, players))
            results[label] = self._smooth(label)
        return results

    # ------------------------------------------------------------------ #
    # Per-frame analysis
    # ------------------------------------------------------------------ #

    def _analyse_team(self, team: str, players: Sequence[PitchCoordinate]) -> dict:
        """Raw (single-frame) formation read for one team."""
        if not players:
            return _empty_result(team)

        # --- direction: the extreme player closest to a goal is the keeper ---
        xs = [p.x for p in players]
        leftmost = players[xs.index(min(xs))]
        rightmost = players[xs.index(max(xs))]
        distance_to_left_goal = leftmost.x
        distance_to_right_goal = self.pitch_length - rightmost.x
        if distance_to_left_goal <= distance_to_right_goal:
            own_goal_x, direction, keeper = 0.0, 1.0, leftmost
        else:
            own_goal_x, direction, keeper = self.pitch_length, -1.0, rightmost

        def progress(p: PitchCoordinate) -> float:
            """Distance from the own goal line along the attacking direction."""
            return (p.x - own_goal_x) * direction

        outfield = [p for p in players if p is not keeper]
        if len(outfield) >= MIN_OUTFIELD_PLAYERS:
            formation, confidence = self._classify_lines(outfield, progress)
        else:
            formation, confidence = "UNKNOWN", 0.0

        width, depth, avg_dist, compactness = self._metrics(players, progress)
        return {
            "team": team,
            "formation": formation,
            "confidence": confidence,
            "width": width,
            "depth": depth,
            "compactness": compactness,
            "avg_teammate_distance": avg_dist,
            "players": len(players),
        }

    def _classify_lines(
        self,
        outfield: Sequence[PitchCoordinate],
        progress,
    ) -> Tuple[str, float]:
        """
        Split outfield players into three lines and match a formation template.

        Returns:
            (formation label, confidence in [0, 1]). The label is ``UNKNOWN``
            when the best match falls below ``min_confidence``.
        """
        ordered = sorted(outfield, key=progress)
        prog = [progress(p) for p in ordered]
        n = len(prog)
        gaps = [prog[i + 1] - prog[i] for i in range(n - 1)]
        # Two largest gaps = the boundaries between the three lines. Ties are
        # broken towards the earlier gap so the split stays deterministic.
        cuts = sorted(sorted(range(len(gaps)), key=lambda i: (-gaps[i], i))[:2])
        band_a = prog[: cuts[0] + 1]
        band_b = prog[cuts[0] + 1 : cuts[1] + 1]
        band_c = prog[cuts[1] + 1 :]
        counts = (len(band_a), len(band_b), len(band_c))

        # --- template match -------------------------------------------------
        # A missing/extra visible player is normal, so partial matches are
        # allowed; the divisor keeps the penalty proportional to team size.
        divisor = max(n, 10)
        best_name, best_conf = "UNKNOWN", 0.0
        for name, template in FORMATION_TEMPLATES.items():
            diff = sum(abs(c - t) for c, t in zip(counts, template))
            conf = 1.0 - diff / divisor
            if conf > best_conf:
                best_name, best_conf = name, conf
        best_conf = max(0.0, min(1.0, best_conf))

        # --- band separation ------------------------------------------------
        # Clean, well-separated lines -> near 1; a uniform blob (no real
        # lines) -> near 0.5 or lower and therefore UNKNOWN.
        gap_min = min(gaps[c] for c in cuts)
        max_spread = max(
            (max(band) - min(band)) for band in (band_a, band_b, band_c)
        )
        denominator = gap_min + max_spread
        separation = (gap_min / denominator) if denominator > 0 else 0.0

        confidence = best_conf * separation
        if confidence < self.min_confidence:
            return "UNKNOWN", confidence
        return best_name, confidence

    def _metrics(
        self,
        players: Sequence[PitchCoordinate],
        progress,
    ) -> Tuple[float, float, float, float]:
        """
        ``(width, depth, average teammate distance, compactness)`` in metres.

        Computed over *every* visible team player - including the goalkeeper -
        so Stage 7 (:mod:`app.team_graph`), which draws the goalkeeper as a
        normal node, reports the identical average teammate distance. The
        keeper is excluded only from the three-line split, where it would
        otherwise anchor the defensive line.
        """
        if not players:
            return 0.0, 0.0, 0.0, 0.0
        ys = [p.y for p in players]
        progs = [progress(p) for p in players]
        width = max(ys) - min(ys)
        depth = max(progs) - min(progs)
        avg_dist, _max_dist = teammate_distance_stats([(p.x, p.y) for p in players])
        compactness = 1.0 / (avg_dist + _COMPACTNESS_FLOOR) if len(players) >= 2 else 0.0
        return width, depth, avg_dist, compactness

    # ------------------------------------------------------------------ #
    # Temporal smoothing
    # ------------------------------------------------------------------ #

    def _smooth(self, team: str) -> dict:
        """Majority vote over the team's sliding window of raw results."""
        window = list(self._window[team])
        if not window:
            return _empty_result(team)

        # Newest first, so equal counts tie-break towards the latest label.
        newest_first = [entry["formation"] for entry in reversed(window)]
        counts = Counter(newest_first)
        mode = max(counts, key=lambda key: counts[key])

        matching = [entry for entry in window if entry["formation"] == mode]
        agreement = len(matching) / len(window)
        mean_confidence = sum(e["confidence"] for e in matching) / len(matching)
        confidence = agreement * mean_confidence
        formation = mode if confidence >= self.min_confidence else "UNKNOWN"

        def mean(key: str) -> float:
            return sum(entry[key] for entry in window) / len(window)

        return {
            "team": team,
            "formation": formation,
            "confidence": confidence,
            "width": mean("width"),
            "depth": mean("depth"),
            "compactness": mean("compactness"),
            "avg_teammate_distance": mean("avg_teammate_distance"),
            "players": window[-1]["players"],
        }
