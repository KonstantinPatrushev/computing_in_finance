"""Tests for the Tabu-based selection solver."""

from __future__ import annotations

import numpy as np
import pytest

from cif.classical.milp import solve_miqp_scip
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_tabu_selection


def _make_problem(n: int = 20, K: int = 5, seed: int = 42) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=K,
    )


def test_tabu_returns_feasible_cardinality_K_subset():
    """Tabu must always honour the requested cardinality."""
    problem = _make_problem(n=15, K=4)
    sol = solve_with_tabu_selection(
        problem, cardinality=4, num_reads=50, tenure=10, seed=1
    )
    assert sol.feasible, "Tabu produced an infeasible portfolio"
    assert len(sol.solver_meta["subset"]) == 4
    assert abs(sol.weights.sum() - 1.0) < 1e-6
    # support must coincide with the reported subset
    support = set(int(i) for i in np.flatnonzero(sol.weights > 1e-9))
    assert support == set(sol.solver_meta["subset"])


def test_tabu_closes_gap_to_scip_on_small_instance():
    """On a 30-asset instance where SCIP solves quickly, Tabu must reach
    near-optimal (≤ 1% gap)."""
    problem = _make_problem(n=30, K=8, seed=7)
    scip = solve_miqp_scip(problem, time_limit_s=30.0)
    tabu = solve_with_tabu_selection(
        problem, cardinality=8, num_reads=200, tenure=20, seed=7
    )
    assert scip.feasible and tabu.feasible
    gap = abs(tabu.objective - scip.objective) / max(abs(scip.objective), 1e-9)
    assert gap < 0.01, f"Tabu gap to SCIP = {gap*100:.2f}% (expected <1%)"


def test_tabu_warm_start_keeps_previous_subset_when_no_improvement():
    """With a high turnover cost and a warm-start subset, Tabu should
    re-pick the warm-start if a fresh search doesn't beat it by enough."""
    problem = _make_problem(n=30, K=4, seed=3)
    warm = [0, 1, 2, 3]
    # Build a previous weights vector matching the warm subset (uniform).
    prev = np.zeros(problem.n)
    prev[warm] = 1.0 / 4
    sol = solve_with_tabu_selection(
        problem,
        cardinality=4,
        num_reads=200,
        tenure=20,
        seed=99,
        warm_start_subset=warm,
        prev_weights=prev,
        turnover_cost_bps=1e6,  # massive cost — nothing should beat warm
    )
    assert sol.feasible
    assert set(sol.solver_meta["subset"]) == set(warm)
