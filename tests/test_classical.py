"""Sanity tests for the classical solver stack."""

from __future__ import annotations

import numpy as np
import pytest

from cif.classical.brute_force import brute_force_cardinality_continuous, brute_force_discrete
from cif.classical.continuous import minimum_variance_portfolio, solve_continuous_mvo
from cif.classical.milp import solve_miqp_ecos_bb, solve_miqp_scip
from cif.problem import PortfolioProblem


def test_n2_minimum_variance_matches_closed_form():
    """For two uncorrelated assets, the analytic min-var weight is sigma2² / (sigma1² + sigma2²)."""
    sigma1_sq, sigma2_sq = 0.04, 0.09
    problem = PortfolioProblem(
        mu=np.array([0.10, 0.15]),
        sigma=np.array([[sigma1_sq, 0.0], [0.0, sigma2_sq]]),
        asset_names=("A", "B"),
    )
    sol = minimum_variance_portfolio(problem)
    expected_w0 = sigma2_sq / (sigma1_sq + sigma2_sq)
    assert sol.feasible
    assert np.isclose(sol.weights[0], expected_w0, atol=1e-4)
    assert np.isclose(sol.weights.sum(), 1.0, atol=1e-6)


def test_continuous_mvo_obeys_box_constraints():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.05, 0.20, size=6)
    A = rng.standard_normal((6, 6))
    sigma = A @ A.T * 0.01 + np.eye(6) * 0.01
    problem = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"A{i}" for i in range(6)),
        w_min=0.0,
        w_max=0.3,
    )
    sol = solve_continuous_mvo(problem)
    assert sol.feasible
    assert np.all(sol.weights >= -1e-6)
    assert np.all(sol.weights <= 0.30 + 1e-6)
    assert np.isclose(sol.weights.sum(), 1.0, atol=1e-6)


def test_brute_force_matches_continuous_for_low_levels(small_problem):
    """At a coarse grid, brute-force discrete should land near the continuous optimum."""
    cont = solve_continuous_mvo(small_problem)
    bf = brute_force_discrete(small_problem, n_levels=20)
    gap = (bf.objective - cont.objective) / abs(cont.objective)
    assert gap < 0.05  # within 5%
    assert np.isclose(bf.weights.sum(), 1.0, atol=1e-6)


def test_brute_force_cardinality_finds_smaller_support():
    rng = np.random.default_rng(2)
    n = 8
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    p = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"X{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=3,
    )
    bf = brute_force_cardinality_continuous(p, cardinality=3)
    n_held = int((bf.weights > 1e-6).sum())
    assert n_held <= 3
    assert np.isclose(bf.weights.sum(), 1.0, atol=1e-6)


def test_scip_and_ecos_bb_agree_on_optimum(medium_problem):
    """Both MIQP backends should reach the same objective on a small instance."""
    scip = solve_miqp_scip(medium_problem)
    ecos = solve_miqp_ecos_bb(medium_problem)
    assert scip.feasible and ecos.feasible
    assert abs(scip.objective - ecos.objective) < 1e-3
    assert scip.solver_meta["n_nonzero"] <= medium_problem.cardinality
    assert ecos.solver_meta["n_nonzero"] <= medium_problem.cardinality
