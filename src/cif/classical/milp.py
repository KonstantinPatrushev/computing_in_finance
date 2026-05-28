"""Mixed-integer quadratic programming (MIQP) baselines.

Two open-source MIQP solvers are wired here — both accessible through cvxpy:

* **ECOS_BB** — simple branch-and-bound on top of ECOS (conic QP). Pure
  Python, bundled with cvxpy, no extra installs. Serves as the "vanilla"
  baseline. Known to be slow for ``N ≥ 30``.
* **SCIP** — production-grade open-source MIP solver with quadratic support
  via ``pyscipopt``. Much faster than ECOS_BB, routinely used in industry as
  the free alternative to Gurobi/CPLEX.

Both are called with the same formulation::

    minimize   − μᵀw + (λ/2) wᵀΣw
    subject to sum(w)       = budget
               w_min · z_i ≤ w_i ≤ w_max · z_i      ∀ i
               sum(z)      ≤ cardinality
               z_i          ∈ {0, 1}

The indicator ``z_i`` activates position ``i``. ``w_min = 0`` is assumed by
default (long-only), which is the standard assumption for the constrained
mean-variance literature and matches the QUBO formulation used elsewhere in
the project.

``time_limit_s`` is enforced in a solver-specific way so Experiment B
(quality-vs-budget) can fairly compare against quantum-inspired SA.
"""

from __future__ import annotations

import time
import warnings

import cvxpy as cp
import numpy as np

from cif.problem import PortfolioProblem, Solution


SolverName = str


def _build_miqp(
    problem: PortfolioProblem,
    cardinality: int | None,
) -> tuple[cp.Problem, cp.Variable, cp.Variable]:
    n = problem.n
    w = cp.Variable(n, name="w")
    z = cp.Variable(n, boolean=True, name="z")

    sigma_psd = cp.psd_wrap(problem.sigma)
    objective = cp.Minimize(
        -problem.mu @ w + 0.5 * problem.risk_aversion * cp.quad_form(w, sigma_psd)
    )
    constraints: list = [
        cp.sum(w) == problem.budget,
        w >= problem.w_min * z,
        w <= problem.w_max * z,
    ]
    if cardinality is not None:
        constraints.append(cp.sum(z) <= cardinality)
    return cp.Problem(objective, constraints), w, z


def _solve_with_kwargs(
    problem_cp: cp.Problem,
    solver: SolverName,
    time_limit_s: float | None,
    verbose: bool,
) -> None:
    kwargs: dict = {"solver": solver, "verbose": verbose}
    if time_limit_s is not None:
        if solver == "SCIP":
            kwargs["scip_params"] = {"limits/time": float(time_limit_s)}
        elif solver == "ECOS_BB":
            # ECOS_BB has no wall-time limit; we pass mi_max_iters as a proxy.
            kwargs["mi_max_iters"] = max(1000, int(time_limit_s * 1000))
    try:
        problem_cp.solve(**kwargs)
    except Exception as exc:
        warnings.warn(f"{solver} solve raised {type(exc).__name__}: {exc}", stacklevel=2)


def solve_miqp(
    problem: PortfolioProblem,
    solver: SolverName = "SCIP",
    cardinality: int | None = None,
    time_limit_s: float | None = None,
    verbose: bool = False,
) -> Solution:
    """Solve the cardinality-constrained MIQP with the named open-source solver.

    Parameters
    ----------
    solver:
        ``"SCIP"`` (recommended) or ``"ECOS_BB"`` (slower, bundled).
    cardinality:
        If not ``None``, use this value. Otherwise falls back to
        ``problem.cardinality``.
    time_limit_s:
        Wall clock time budget. SCIP respects this via a native parameter;
        ECOS_BB uses ``mi_max_iters`` as a proxy.
    """
    effective_card = cardinality if cardinality is not None else problem.cardinality
    cp_problem, w, z = _build_miqp(problem, effective_card)

    if solver not in cp.installed_solvers():
        raise RuntimeError(
            f"Solver {solver!r} not available. Installed: {cp.installed_solvers()}"
        )

    t0 = time.perf_counter()
    _solve_with_kwargs(cp_problem, solver, time_limit_s, verbose)
    wall = time.perf_counter() - t0

    if w.value is None:
        return Solution(
            weights=np.zeros(problem.n),
            objective=np.inf,
            feasible=False,
            wall_time_s=wall,
            solver=f"cvxpy_miqp/{solver}",
            solver_meta={
                "status": cp_problem.status,
                "cardinality": effective_card,
                "time_limit_s": time_limit_s,
                "time_limit_hit": True,
            },
        )

    weights = np.asarray(w.value, dtype=float)
    weights = np.where(weights < 1e-9, 0.0, weights)
    obj = problem.objective_value(weights)

    budget_ok = abs(weights.sum() - problem.budget) < 1e-4
    box_ok = bool(np.all((weights >= problem.w_min - 1e-6) & (weights <= problem.w_max + 1e-6)))
    card_ok = True
    if effective_card is not None:
        card_ok = int((weights > 1e-6).sum()) <= effective_card
    feasible = budget_ok and box_ok and card_ok

    return Solution(
        weights=weights,
        objective=obj,
        feasible=feasible,
        wall_time_s=wall,
        solver=f"cvxpy_miqp/{solver}",
        solver_meta={
            "status": cp_problem.status,
            "cvxpy_value": float(cp_problem.value) if cp_problem.value is not None else None,
            "cardinality": effective_card,
            "time_limit_s": time_limit_s,
            "n_nonzero": int((weights > 1e-6).sum()),
            "budget_ok": budget_ok,
            "box_ok": box_ok,
            "card_ok": card_ok,
        },
    )


def solve_miqp_scip(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    time_limit_s: float | None = None,
    verbose: bool = False,
) -> Solution:
    """Convenience wrapper for SCIP, the default MIQP solver."""
    return solve_miqp(
        problem,
        solver="SCIP",
        cardinality=cardinality,
        time_limit_s=time_limit_s,
        verbose=verbose,
    )


def solve_miqp_ecos_bb(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    time_limit_s: float | None = None,
    verbose: bool = False,
) -> Solution:
    """Convenience wrapper for ECOS_BB, the vanilla bundled baseline."""
    return solve_miqp(
        problem,
        solver="ECOS_BB",
        cardinality=cardinality,
        time_limit_s=time_limit_s,
        verbose=verbose,
    )
