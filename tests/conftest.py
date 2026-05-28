"""Shared fixtures for cif tests."""

from __future__ import annotations

import numpy as np
import pytest

from cif.problem import PortfolioProblem


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260415)


@pytest.fixture
def small_problem(rng: np.random.Generator) -> PortfolioProblem:
    """Small synthetic 5-asset problem with known PSD covariance."""
    n = 5
    mu = rng.uniform(0.05, 0.20, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.01 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"A{i}" for i in range(n)),
        risk_aversion=2.0,
    )


@pytest.fixture
def medium_problem(rng: np.random.Generator) -> PortfolioProblem:
    n = 12
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"M{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=4,
    )
