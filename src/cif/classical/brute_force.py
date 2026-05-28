"""Exhaustive enumeration for discrete Markowitz ground truth (N ≤ 12).

We enumerate **integer compositions** of a granularity parameter ``n_levels``
into ``N`` buckets — every integer tuple ``(k_0, …, k_{N-1})`` with
``sum(k) = n_levels`` maps to a feasible discrete weight vector
``w_i = k_i / n_levels * budget``.

Why compositions and not unconstrained binary strings?

1. The budget constraint ``sum(w) = budget`` is enforced **structurally**:
   no wasted evaluations on infeasible combinations. This is strictly cheaper
   than enumerating all ``2^(N·B)`` binary strings and then filtering.
2. It matches the natural "discrete Markowitz with fixed granularity"
   interpretation used throughout the report.

This is **not** an encoding that mirrors the QUBO one-to-one. The QUBO binary
formulation adds budget-violation as a penalty rather than enforcing it
exactly; the brute-force enumerator here deliberately restricts to the
feasible manifold to give a **certified discrete optimum**.

Size table (no cardinality constraint):

========  ============  =========================
  N        n_levels       number of compositions
========  ============  =========================
   5          10                      2 002
   8          10                     24 310
  10          10                     92 378
  12          10                    352 716
  10          20                 10 015 005
  12          20                300 540 195   ← too many
========  ============  =========================

Cardinality constraint ``|support(w)| ≤ K`` reduces the count further by
restricting which subsets of size up to ``K`` are allowed to hold mass.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from itertools import combinations

import numpy as np

from cif.problem import PortfolioProblem, Solution


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every non-negative integer tuple of length ``parts`` summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, parts - 1):
            yield (first,) + tail


def brute_force_discrete(
    problem: PortfolioProblem,
    n_levels: int = 10,
    time_budget_s: float | None = None,
) -> Solution:
    """Enumerate all discrete weight vectors at the given granularity.

    Parameters
    ----------
    n_levels:
        Number of discrete steps across ``[0, budget]``; weight granularity
        is ``budget / n_levels``.
    time_budget_s:
        If not None, abort after this many seconds and return the best
        feasible solution seen so far (with ``feasible=True`` and
        ``solver_meta["aborted_by_budget"]=True``).
    """
    n = problem.n
    step = problem.budget / n_levels
    if step < problem.w_min - 1e-12:
        raise ValueError(
            f"step {step} smaller than w_min {problem.w_min}; increase n_levels"
        )

    w_max_ticks = int(np.floor(problem.w_max / step + 1e-9))
    w_min_ticks = int(np.ceil(problem.w_min / step - 1e-9))

    best_w = None
    best_obj = np.inf
    visited = 0
    t0 = time.perf_counter()

    for comp in _compositions(n_levels, n):
        visited += 1
        # Box constraint per asset
        if any(k > w_max_ticks or k < w_min_ticks for k in comp):
            continue
        # Cardinality constraint
        if problem.cardinality is not None:
            if sum(1 for k in comp if k > 0) > problem.cardinality:
                continue
        weights = np.asarray(comp, dtype=float) * step
        obj = problem.objective_value(weights)
        if obj < best_obj:
            best_obj = obj
            best_w = weights
        if time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s:
            aborted = True
            break
    else:
        aborted = False

    wall = time.perf_counter() - t0

    if best_w is None:
        raise RuntimeError("Brute force did not find any feasible solution")

    return Solution(
        weights=best_w,
        objective=best_obj,
        feasible=True,
        wall_time_s=wall,
        solver=f"brute_force/n_levels={n_levels}",
        solver_meta={
            "visited": visited,
            "n_levels": n_levels,
            "aborted_by_budget": aborted,
        },
    )


def brute_force_cardinality_continuous(
    problem: PortfolioProblem,
    cardinality: int,
) -> Solution:
    """Upper-bound ground truth: for every size-K subset, solve continuous MVO.

    For each subset ``S`` of ``K`` assets, the sub-problem is convex and has
    a cvxpy solution in milliseconds. Returns the best subset.

    Useful as a sanity check when you want the **continuous** optimum of
    cardinality-constrained Markowitz (no weight discretisation), e.g. for
    comparing against MIQP solvers.
    """
    from cif.classical.continuous import solve_continuous_mvo

    n = problem.n
    if not (1 <= cardinality <= n):
        raise ValueError(f"cardinality must be in [1, {n}], got {cardinality}")

    best_sol: Solution | None = None
    best_obj = np.inf
    evaluated = 0
    t0 = time.perf_counter()

    for idx in combinations(range(n), cardinality):
        idx_arr = np.asarray(idx)
        sub_mu = problem.mu[idx_arr]
        sub_sigma = problem.sigma[np.ix_(idx_arr, idx_arr)]
        sub_names = tuple(problem.asset_names[i] for i in idx)
        sub_problem = PortfolioProblem(
            mu=sub_mu,
            sigma=sub_sigma,
            asset_names=sub_names,
            w_min=problem.w_min,
            w_max=problem.w_max,
            budget=problem.budget,
            risk_aversion=problem.risk_aversion,
        )
        try:
            sol = solve_continuous_mvo(sub_problem)
        except RuntimeError:
            continue
        full_weights = np.zeros(n, dtype=float)
        full_weights[idx_arr] = sol.weights
        obj = problem.objective_value(full_weights)
        evaluated += 1
        if obj < best_obj:
            best_obj = obj
            best_sol = Solution(
                weights=full_weights,
                objective=obj,
                feasible=True,
                wall_time_s=0.0,  # overwritten below
                solver="brute_force_cardinality_continuous",
                solver_meta={"support": list(idx), "inner_solver": sol.solver},
            )

    wall = time.perf_counter() - t0
    if best_sol is None:
        raise RuntimeError("Cardinality brute force found no feasible subset")
    best_sol.wall_time_s = wall
    best_sol.solver_meta["subsets_evaluated"] = evaluated
    return best_sol
