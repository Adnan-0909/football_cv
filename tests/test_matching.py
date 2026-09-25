"""
Unit Tests for the Linear Assignment Module
===========================================

Validates the in-house Hungarian solver against a brute-force optimum, plus the
threshold wrapper used by the tracker.
"""

import itertools

import numpy as np
import pytest

from app.matching import linear_assignment, linear_sum_assignment


def brute_force_min_cost(cost: np.ndarray) -> float:
    """Minimum total assignment cost by exhaustive search (small matrices only)."""
    n_rows, n_cols = cost.shape
    if n_rows <= n_cols:
        assignments = (
            (range(n_rows), perm)
            for perm in itertools.permutations(range(n_cols), n_rows)
        )
    else:
        assignments = (
            (perm, range(n_cols))
            for perm in itertools.permutations(range(n_rows), n_cols)
        )
    return min(
        sum(cost[r, c] for r, c in zip(rows, cols)) for rows, cols in assignments
    )


def assignment_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(cost[rows, cols].sum())


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4), (3, 5), (5, 3), (2, 6), (6, 2)])
def test_matches_brute_force_optimum(shape):
    rng = np.random.default_rng(42)
    for _ in range(25):
        cost = rng.random(shape)
        rows, cols = linear_sum_assignment(cost)
        expected = brute_force_min_cost(cost)
        assert assignment_cost(cost, rows, cols) == pytest.approx(expected)


def test_rows_and_cols_are_one_to_one():
    rng = np.random.default_rng(0)
    cost = rng.random((7, 7))
    rows, cols = linear_sum_assignment(cost)
    assert sorted(rows.tolist()) == list(range(7))
    assert len(set(cols.tolist())) == 7


def test_rectangular_matrix_leaves_surplus_unassigned():
    cost = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.7]])
    rows, cols = linear_sum_assignment(cost)
    assert len(rows) == 2
    assert len(set(rows.tolist())) == 2
    assert len(set(cols.tolist())) == 2
    # Surplus column is the one left out of the optimal pairing.
    assert set(cols.tolist()) == {0, 1}


def test_empty_and_degenerate_shapes():
    for shape in [(0, 0), (0, 4), (4, 0)]:
        rows, cols = linear_sum_assignment(np.empty(shape))
        assert rows.size == 0 and cols.size == 0


def test_non_finite_costs_are_rejected():
    with pytest.raises(ValueError):
        linear_sum_assignment(np.array([[1.0, np.nan], [0.2, 0.3]]))


def test_linear_assignment_rejects_pairs_above_threshold():
    # Optimal pairing is (0,0) cost 0.1 and (1,1) cost 0.9, but 0.9 > 0.5 so
    # only the first pair survives the threshold.
    cost = np.array([[0.1, 0.95], [0.9, 0.9]])
    matches, u_rows, u_cols = linear_assignment(cost, thresh=0.5)
    assert matches.tolist() == [[0, 0]]
    assert u_rows.tolist() == [1]
    assert u_cols.tolist() == [1]


def test_linear_assignment_on_empty_matrix():
    matches, u_rows, u_cols = linear_assignment(np.empty((0, 3)), thresh=0.5)
    assert matches.shape == (0, 2)
    assert u_rows.size == 0
    assert u_cols.tolist() == [0, 1, 2]
