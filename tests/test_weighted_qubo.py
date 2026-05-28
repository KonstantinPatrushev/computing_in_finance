"""Tests for the weighted-binary-encoding QUBO."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from cif.problem import PortfolioProblem
from cif.qubo.builder import build_weighted_bqm, sample_to_weights_weighted


def _make_problem(n: int = 3, seed: int = 11) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.20, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.01 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
    )


def test_weighted_bqm_energy_matches_objective_on_feasible_states():
    """For every weighted-encoding state that satisfies Σw = 1, the BQM
    energy must equal the Markowitz objective on the decoded weights."""
    problem = _make_problem(n=3)
    B = 2
    bqm = build_weighted_bqm(problem, bits_per_asset=B, budget_penalty=0.0)

    n = problem.n
    keys = [(i, k) for i in range(n) for k in range(B)]
    max_err = 0.0
    feasible_count = 0
    for bits in product((0, 1), repeat=n * B):
        sample = dict(zip(keys, bits))
        weights = sample_to_weights_weighted(sample, n, B, renormalize=False)
        if abs(weights.sum() - 1.0) > 1e-9:
            continue
        energy = bqm.energy(sample)
        obj = problem.objective_value(weights)
        max_err = max(max_err, abs(energy - obj))
        feasible_count += 1

    assert feasible_count > 0, "no budget-feasible states found"
    assert max_err < 1e-9, f"max |energy - objective| = {max_err:g}"


def test_weighted_bqm_budget_penalty_dominates():
    """Setting a high budget penalty must put any Σw ≠ 1 state strictly
    above any Σw = 1 state."""
    problem = _make_problem(n=3, seed=7)
    B = 2
    bqm = build_weighted_bqm(problem, bits_per_asset=B, budget_penalty=100.0)

    n = problem.n
    keys = [(i, k) for i in range(n) for k in range(B)]
    best_feasible = float("inf")
    best_infeasible = float("inf")
    for bits in product((0, 1), repeat=n * B):
        sample = dict(zip(keys, bits))
        weights = sample_to_weights_weighted(sample, n, B, renormalize=False)
        energy = bqm.energy(sample)
        if abs(weights.sum() - 1.0) < 1e-9:
            best_feasible = min(best_feasible, energy)
        else:
            best_infeasible = min(best_infeasible, energy)

    assert best_feasible < best_infeasible, (
        f"budget penalty too weak: feasible {best_feasible} >= infeasible {best_infeasible}"
    )


def test_sample_to_weights_weighted_renormalisation_preserves_sum_one():
    """When `renormalize=True`, decoded weights must sum to 1 (or be zero)."""
    n, B = 4, 3
    rng = np.random.default_rng(0)
    for _ in range(20):
        bits = rng.integers(0, 2, size=n * B).tolist()
        sample = {(i, k): int(bits[i * B + k]) for i in range(n) for k in range(B)}
        w = sample_to_weights_weighted(sample, n, B, renormalize=True)
        s = float(w.sum())
        assert s == 0.0 or abs(s - 1.0) < 1e-12, f"sum = {s}"
