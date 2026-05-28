"""Tests for QUBO builders."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from cif.classical.brute_force import brute_force_discrete
from cif.qubo.builder import (
    build_bqm,
    build_selection_bqm,
    sample_to_weights,
    selection_bits_to_subset,
    suggest_penalties,
)
from cif.qubo.encoding import make_encoding


def test_weighted_bqm_energy_matches_objective_on_feasible_states(small_problem):
    """For every (one-hot, budget-feasible) state, BQM energy must equal objective."""
    enc = make_encoding(n_assets=small_problem.n, n_levels=4, kind="one_hot")
    pen = suggest_penalties(small_problem, enc, multiplier=10.0)
    bqm = build_bqm(small_problem, enc, pen)

    L = enc.n_levels
    feasible_count = 0
    for levels in product(range(L + 1), repeat=small_problem.n):
        if sum(levels) != L:
            continue
        sample = {
            (i, k): (1 if levels[i] == k else 0)
            for i in range(small_problem.n)
            for k in range(L + 1)
        }
        weights = sample_to_weights(sample, enc)
        energy = bqm.energy(sample)
        obj = small_problem.objective_value(weights)
        assert abs(energy - obj) < 1e-9, f"levels {levels}: energy {energy} != obj {obj}"
        feasible_count += 1
    assert feasible_count > 0


def test_weighted_bqm_minimum_matches_brute_force(small_problem):
    enc = make_encoding(n_assets=small_problem.n, n_levels=4, kind="one_hot")
    pen = suggest_penalties(small_problem, enc, multiplier=10.0)
    bqm = build_bqm(small_problem, enc, pen)
    bf = brute_force_discrete(small_problem, n_levels=4)

    # Enumerate every one-hot/budget-feasible state and pick the lowest energy
    L = enc.n_levels
    best_energy = float("inf")
    best_weights: np.ndarray | None = None
    for levels in product(range(L + 1), repeat=small_problem.n):
        if sum(levels) != L:
            continue
        sample = {
            (i, k): (1 if levels[i] == k else 0)
            for i in range(small_problem.n)
            for k in range(L + 1)
        }
        e = bqm.energy(sample)
        if e < best_energy:
            best_energy = e
            best_weights = sample_to_weights(sample, enc)

    assert best_weights is not None
    assert abs(bf.objective - best_energy) < 1e-9, "BQM minimum disagrees with brute force"
    assert np.allclose(bf.weights, best_weights)


def test_selection_bqm_cardinality_penalty_dominates():
    """The cardinality penalty should make off-K subsets strictly worse."""
    n = 6
    K = 3
    rng = np.random.default_rng(11)
    mu = rng.uniform(0.05, 0.20, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.01 + np.eye(n) * 0.01
    from cif.problem import PortfolioProblem  # local import to avoid fixture clash

    problem = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=K,
    )
    bqm = build_selection_bqm(problem, cardinality=K)

    # Best size-K subset
    best_K = float("inf")
    for subset in _all_subsets(n, K):
        sample = {i: (1 if i in subset else 0) for i in range(n)}
        e = bqm.energy(sample)
        best_K = min(best_K, e)

    # Best size-(K+1) subset — should be strictly worse due to penalty
    best_K1 = float("inf")
    for subset in _all_subsets(n, K + 1):
        sample = {i: (1 if i in subset else 0) for i in range(n)}
        e = bqm.energy(sample)
        best_K1 = min(best_K1, e)

    assert best_K < best_K1, f"cardinality penalty too weak: {best_K} vs {best_K1}"


def test_selection_bits_to_subset_dict_and_array_agree():
    sample_dict = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1}
    sample_array = np.array([0, 1, 0, 1, 1])
    assert selection_bits_to_subset(sample_dict, 5) == [1, 3, 4]
    assert selection_bits_to_subset(sample_array, 5) == [1, 3, 4]


def _all_subsets(n: int, k: int):
    from itertools import combinations
    return list(combinations(range(n), k))
