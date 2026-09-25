"""
Linear Assignment Module
========================

Pure NumPy implementation of the linear assignment problem (Hungarian /
Jonker-Volgenant shortest-augmenting-path algorithm).

This module exists so the tracker can perform optimal detection <-> track
association without pulling in extra dependencies:

- ``scipy.optimize.linear_sum_assignment`` is not installed in this project.
- ``lap`` (used by Ultralytics' own ByteTracker) has no reliable wheels for
  Windows + Python 3.12 and would trigger an auto ``pip install`` at runtime.

Cost matrices here are small (typically <= 40x40: ~22 players + ball), so a
Python-level augmenting loop is more than fast enough for real-time use.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the rectangular linear assignment problem (minimize total cost).

    Args:
        cost_matrix: 2-D array of shape (n_rows, n_cols) with finite costs.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            row_ind - indices of assigned rows, sorted ascending.
            col_ind - matching column index for each entry of row_ind.
            If one dimension is larger, the surplus rows/columns are simply
            left unassigned (same convention as SciPy).
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost_matrix must be 2-D, got shape {cost.shape}")
    if cost.size and not np.all(np.isfinite(cost)):
        raise ValueError("cost_matrix must contain only finite values")

    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # The algorithm assumes rows <= cols; solve the transpose when it does not.
    transposed = n_rows > n_cols
    if transposed:
        cost = cost.T
    n, m = cost.shape

    # 1-based working table; row 0 and column 0 are sentinels only.
    table = np.zeros((n + 1, m + 1), dtype=np.float64)
    table[1:, 1:] = cost

    u = np.zeros(n + 1, dtype=np.float64)  # row potentials
    v = np.zeros(m + 1, dtype=np.float64)  # column potentials
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row assigned to column j (0 = free)
    way = np.zeros(m + 1, dtype=np.int64)  # augmenting path bookkeeping

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = int(p[j0])
            free = ~used[1:]
            candidate = table[i0, 1:] - u[i0] - v[1:]

            improved = free & (candidate < minv[1:])
            minv[1:][improved] = candidate[improved]
            way[1:][improved] = j0

            # Column with the smallest reduced cost among the free ones.
            free_minv = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(free_minv)) + 1
            delta = float(free_minv[j1 - 1])
            if not np.isfinite(delta):
                raise RuntimeError("Assignment failed: no free column available")

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Walk the augmenting path and flip the assignments along it.
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    cols = np.array([j for j in range(1, m + 1) if p[j] != 0], dtype=np.int64) - 1
    rows = np.array([p[j] for j in range(1, m + 1) if p[j] != 0], dtype=np.int64) - 1

    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows, kind="stable")
    return rows[order], cols[order]


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign rows to columns, discarding pairs whose cost exceeds ``thresh``.

    Mirrors Ultralytics/ByteTrack's ``matching.linear_assignment`` so callers
    can keep using their familiar threshold semantics.

    Args:
        cost_matrix: 2-D array of shape (n_tracks, n_detections).
        thresh: Maximum acceptable cost for a match (cost = 1 - similarity).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            matches - (K, 2) int array of (track_index, detection_index) pairs.
            unmatched_tracks - indices of rows without an accepted match.
            unmatched_dets - indices of columns without an accepted match.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if cost.size == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(n_rows, dtype=np.int64),
            np.arange(n_cols, dtype=np.int64),
        )

    rows, cols = linear_sum_assignment(cost)
    accepted = cost[rows, cols] <= thresh
    matches = (
        np.stack([rows[accepted], cols[accepted]], axis=1)
        if np.any(accepted)
        else np.empty((0, 2), dtype=np.int64)
    )

    matched_rows = set(matches[:, 0].tolist())
    matched_cols = set(matches[:, 1].tolist())
    unmatched_rows = np.array([i for i in range(n_rows) if i not in matched_rows], dtype=np.int64)
    unmatched_cols = np.array([j for j in range(n_cols) if j not in matched_cols], dtype=np.int64)
    return matches, unmatched_rows, unmatched_cols
