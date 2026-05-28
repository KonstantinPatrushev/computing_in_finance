"""Core problem and solution types shared by classical and quantum solvers.

All solvers in the project consume a :class:`PortfolioProblem` and return one
or more :class:`Solution` objects so that metrics and visualisation can be
computed uniformly, regardless of which backend produced the weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PortfolioProblem:
    """Input specification of a discrete Markowitz instance.

    Attributes
    ----------
    mu:
        Annualised expected-return vector, shape ``(N,)``.
    sigma:
        Annualised covariance matrix, shape ``(N, N)``, positive semidefinite.
    asset_names:
        Human-readable labels, length ``N``. Kept for report figures.
    w_min, w_max:
        Lower and upper bound on each weight. Continuous solvers enforce the
        range directly; discrete encoders use the range as the binary grid.
    budget:
        ``sum(w)`` target. Almost always ``1.0``.
    risk_aversion:
        Coefficient ``λ`` in the mean-variance objective
        ``max μᵀw − (λ/2) wᵀΣw``. Higher ``λ`` means more risk averse.
    cardinality:
        Maximum number of non-zero positions allowed (``None`` — no
        cardinality constraint).
    """

    mu: np.ndarray
    sigma: np.ndarray
    asset_names: tuple[str, ...]
    w_min: float = 0.0
    w_max: float = 1.0
    budget: float = 1.0
    risk_aversion: float = 1.0
    cardinality: int | None = None

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if mu.ndim != 1:
            raise ValueError(f"mu must be 1D, got shape {mu.shape}")
        if sigma.shape != (mu.shape[0], mu.shape[0]):
            raise ValueError(f"sigma shape {sigma.shape} incompatible with mu {mu.shape}")
        if len(self.asset_names) != mu.shape[0]:
            raise ValueError(
                f"asset_names length {len(self.asset_names)} != N={mu.shape[0]}"
            )
        if self.w_min > self.w_max:
            raise ValueError(f"w_min={self.w_min} > w_max={self.w_max}")
        if self.cardinality is not None and not (1 <= self.cardinality <= mu.shape[0]):
            raise ValueError(
                f"cardinality must be in [1, N={mu.shape[0]}], got {self.cardinality}"
            )
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    @property
    def n(self) -> int:
        """Number of assets."""
        return self.mu.shape[0]

    def objective_value(self, weights: np.ndarray) -> float:
        """Evaluate the mean-variance objective at ``weights``.

        We minimise ``-μᵀw + (λ/2) wᵀΣw`` so that lower is better; this keeps
        the convention aligned with QUBO/Ising energy minimisation downstream.
        """
        w = np.asarray(weights, dtype=float)
        linear = -float(self.mu @ w)
        quadratic = 0.5 * self.risk_aversion * float(w @ self.sigma @ w)
        return linear + quadratic


@dataclass
class Solution:
    """Result of a solver run.

    The ``feasible`` flag reflects whether the returned ``weights`` satisfy
    all declared constraints of the problem (budget, box, cardinality),
    measured with ``rtol`` tolerance by :mod:`cif.metrics.feasibility`.
    """

    weights: np.ndarray
    objective: float
    feasible: bool
    wall_time_s: float
    solver: str
    solver_meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
