"""Wrapper around :class:`dwave.samplers.SimulatedAnnealingSampler`.

The D-Wave ``neal`` sampler runs classical simulated annealing on a BQM. It
is quantum-inspired (the acceptance schedule echoes quantum-annealing
literature) but runs entirely on a CPU — no API keys, no quotas. In this
project it plays the role of the "quantum-inspired" baseline that the report
compares against classical MIQP solvers.
"""

from __future__ import annotations

import time

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler

from cif.classical.continuous import solve_continuous_mvo
from cif.problem import PortfolioProblem, Solution
from cif.qubo.builder import (
    Penalties,
    build_bqm,
    build_selection_bqm,
    sample_to_weights,
    selection_bits_to_subset,
)
from cif.qubo.encoding import WeightEncoding


def solve_with_neal(
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    penalties: Penalties,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int | None = None,
    beta_range: tuple[float, float] | None = None,
    return_all_solutions: bool = False,
) -> Solution | list[Solution]:
    """Run simulated annealing on a QUBO derived from ``problem``.

    Parameters
    ----------
    num_reads:
        Number of independent anneal runs. Each gives one candidate solution;
        the lowest-energy **feasible** sample is returned.
    num_sweeps:
        Monte Carlo sweeps per run. Default 1000 is a conservative sweet spot
        for N·(L+1) ≤ 1000 variables.
    seed:
        RNG seed for reproducibility.
    beta_range:
        Explicit inverse-temperature schedule bounds. ``None`` lets ``neal``
        choose based on the problem's coefficient magnitudes.
    return_all_solutions:
        If True, return a list of decoded solutions sorted by energy instead
        of a single best one.
    """
    bqm = build_bqm(problem, encoding, penalties)

    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_range=beta_range,
    )
    wall = time.perf_counter() - t0

    return _sampleset_to_solutions(
        sampleset=sampleset,
        problem=problem,
        encoding=encoding,
        wall_time_s=wall,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        return_all=return_all_solutions,
    )


def _sampleset_to_solutions(
    sampleset: dimod.SampleSet,
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    wall_time_s: float,
    num_reads: int,
    num_sweeps: int,
    return_all: bool,
) -> Solution | list[Solution]:
    solutions: list[Solution] = []
    for record in sampleset.record:
        sample_dict = {
            var: int(record.sample[i])
            for i, var in enumerate(sampleset.variables)
        }
        weights = sample_to_weights(sample_dict, encoding)
        obj = problem.objective_value(weights)
        feasible = _check_feasible(weights, problem)
        solutions.append(
            Solution(
                weights=weights,
                objective=obj,
                feasible=bool(feasible),
                wall_time_s=wall_time_s / max(num_reads, 1),
                solver=f"neal/SA[{num_reads}x{num_sweeps}]",
                solver_meta={
                    "num_reads": num_reads,
                    "num_sweeps": num_sweeps,
                    "qubo_energy": float(record.energy),
                    "num_occurrences": int(record.num_occurrences),
                },
            )
        )

    if not solutions:
        raise RuntimeError("neal returned an empty sample set")

    if return_all:
        solutions.sort(key=lambda s: (not s.feasible, s.objective))
        return solutions

    feasible_solutions = [s for s in solutions if s.feasible]
    best = (
        min(feasible_solutions, key=lambda s: s.objective)
        if feasible_solutions
        else min(solutions, key=lambda s: s.objective)
    )
    best.wall_time_s = wall_time_s
    best.solver_meta = {
        **best.solver_meta,
        "n_samples": len(solutions),
        "n_feasible": len(feasible_solutions),
    }
    return best


def solve_with_neal_selection(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    num_reads: int = 500,
    num_sweeps: int = 500,
    seed: int | None = None,
    cardinality_penalty: float | None = None,
    beta_range: tuple[float, float] | None = None,
    top_subsets: int = 5,
    refine_continuous: bool = True,
    warm_start_subset: list[int] | tuple[int, ...] | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_cost_bps: float = 0.0,
) -> Solution:
    """Two-stage pipeline: ``neal`` picks a subset, then cvxpy refines weights.

    Stage 1: Build the binary-inclusion BQM (only ``N`` variables) and run
    simulated annealing. Collect the best few unique subsets found.

    Stage 2: For each candidate subset of exactly ``K`` assets, solve the
    continuous mean-variance subproblem via :mod:`cif.classical.continuous`
    (few milliseconds per subset at ``N ≤ 100``). Return the subset whose
    continuous refinement has the lowest full-problem objective.

    This corresponds to how practitioners actually use quantum-inspired
    solvers: delegate the hard combinatorial selection to SA, then do
    cheap convex post-processing. The two-stage timing is what we report.

    Parameters
    ----------
    warm_start_subset:
        If provided, this subset is injected into the candidate list
        (alongside SA's findings) for continuous refinement. The comparison
        keeps ``warm_start_subset`` if it produces a better refined
        objective than any of SA's top-K — essential for walk-forward
        stability, where the previous period's subset is usually a strong
        baseline SA may not rediscover.
    prev_weights:
        If provided together with ``turnover_cost_bps > 0``, the candidate
        ranking uses a turnover-adjusted objective
        ``obj + tc_bps · ||w − prev_weights||_1 / 20000``. This makes the
        fold-to-fold comparison honest about transaction-cost drag.
    turnover_cost_bps:
        Per-rebalance cost assumption in basis points.
    """
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)

    bqm = build_selection_bqm(problem, cardinality=K, cardinality_penalty=cardinality_penalty)

    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_range=beta_range,
    )
    sa_wall = time.perf_counter() - t0

    # Collect unique feasible subsets of exactly size K, ordered by SA energy
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for record in sampleset.record:
        sample_dict = {
            var: int(record.sample[i]) for i, var in enumerate(sampleset.variables)
        }
        subset = tuple(selection_bits_to_subset(sample_dict, problem.n))
        if len(subset) != K:
            continue
        if subset in seen:
            continue
        seen.add(subset)
        candidates.append((float(record.energy), subset))
        if len(candidates) >= top_subsets:
            break

    if warm_start_subset is not None:
        ws = tuple(sorted(int(i) for i in warm_start_subset))
        if len(ws) == K and ws not in seen:
            seen.add(ws)
            candidates.append((float("inf"), ws))

    refine_t0 = time.perf_counter()
    scored: list[tuple[float, Solution]] = []
    evaluated = 0
    for _, subset in candidates:
        if refine_continuous:
            idx = np.asarray(subset)
            sub_problem = PortfolioProblem(
                mu=problem.mu[idx],
                sigma=problem.sigma[np.ix_(idx, idx)],
                asset_names=tuple(problem.asset_names[i] for i in subset),
                w_min=problem.w_min,
                w_max=problem.w_max,
                budget=problem.budget,
                risk_aversion=problem.risk_aversion,
            )
            try:
                sub_sol = solve_continuous_mvo(sub_problem)
            except RuntimeError:
                continue
            weights = np.zeros(problem.n, dtype=float)
            weights[idx] = sub_sol.weights
        else:
            weights = np.zeros(problem.n, dtype=float)
            weights[list(subset)] = 1.0 / K
        obj = problem.objective_value(weights)
        if prev_weights is not None and turnover_cost_bps > 0:
            turnover = 0.5 * float(np.sum(np.abs(weights - prev_weights)))
            cost = turnover_cost_bps * turnover / 10000.0
            ranking_obj = obj + cost
        else:
            ranking_obj = obj
        evaluated += 1
        sol_candidate = Solution(
            weights=weights,
            objective=obj,
            feasible=_check_feasible(weights, problem),
            wall_time_s=0.0,
            solver=(
                f"neal_selection+cvxpy[{num_reads}x{num_sweeps}]"
                if refine_continuous
                else f"neal_selection[{num_reads}x{num_sweeps}]"
            ),
            solver_meta={"subset": list(subset)},
        )
        scored.append((ranking_obj, sol_candidate))
    refine_wall = time.perf_counter() - refine_t0

    best: Solution | None = None
    if scored:
        scored.sort(key=lambda pair: pair[0])
        best = scored[0][1]

    if best is None:
        # Fallback: no SA sample had exactly K actives. Return best-effort.
        first = sampleset.first
        sample_dict = {var: int(first.sample[var]) for var in sampleset.variables}
        subset_any = selection_bits_to_subset(sample_dict, problem.n)
        weights = np.zeros(problem.n, dtype=float)
        if subset_any:
            weights[list(subset_any)] = 1.0 / len(subset_any)
        return Solution(
            weights=weights,
            objective=problem.objective_value(weights),
            feasible=False,
            wall_time_s=sa_wall + refine_wall,
            solver=f"neal_selection[{num_reads}x{num_sweeps}]",
            solver_meta={
                "n_candidates": 0,
                "fallback_subset": list(subset_any),
                "sa_wall_s": sa_wall,
                "refine_wall_s": refine_wall,
            },
        )

    best.wall_time_s = sa_wall + refine_wall
    best.solver_meta.update({
        "sa_wall_s": sa_wall,
        "refine_wall_s": refine_wall,
        "n_candidates": len(candidates),
        "n_evaluated": evaluated,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
    })
    return best


def solve_with_tabu_selection(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    num_reads: int = 200,
    tenure: int = 20,
    seed: int | None = None,
    cardinality_penalty: float | None = None,
    top_subsets: int = 50,
    refine_continuous: bool = True,
    warm_start_subset: list[int] | tuple[int, ...] | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_cost_bps: float = 0.0,
) -> Solution:
    """Two-stage pipeline using TabuSampler instead of simulated annealing.

    Same architecture as :func:`solve_with_neal_selection` (binary-inclusion
    QUBO → top-K subset candidates → continuous refinement via cvxpy on each),
    but the sampler is ``dwave.samplers.TabuSampler``. In our benchmarks Tabu
    closes the 8–13% optimisation gap that SA leaves to under 0.3% on dense
    random covariance matrices at N ≤ 200, at comparable or better wall time.

    The default ``num_reads=200, tenure=20`` configuration is the calibrated
    sweet spot from internal benchmarks (Apr 2026): 4–5 seconds total at
    N up to 200, gap typically ≤ 0.25% vs SCIP MIQP, predictable wall time
    with std/mean ≈ 0.05 across instances.

    Parameters mirror :func:`solve_with_neal_selection` so the two functions
    are drop-in interchangeable in the experiment runners.
    """
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)

    bqm = build_selection_bqm(problem, cardinality=K, cardinality_penalty=cardinality_penalty)

    sampler = TabuSampler()
    # Tabu requires 0 ≤ tenure < num_vars; cap to leave at least one free move.
    effective_tenure = min(tenure, max(1, problem.n - 1))
    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        tenure=effective_tenure,
        seed=seed,
    )
    sample_wall = time.perf_counter() - t0

    # Collect unique feasible subsets ordered by Tabu energy
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for record in sampleset.record:
        sample_dict = {
            var: int(record.sample[i]) for i, var in enumerate(sampleset.variables)
        }
        subset = tuple(selection_bits_to_subset(sample_dict, problem.n))
        if len(subset) != K:
            continue
        if subset in seen:
            continue
        seen.add(subset)
        candidates.append((float(record.energy), subset))
        if len(candidates) >= top_subsets:
            break

    if warm_start_subset is not None:
        ws = tuple(sorted(int(i) for i in warm_start_subset))
        if len(ws) == K and ws not in seen:
            seen.add(ws)
            candidates.append((float("inf"), ws))

    refine_t0 = time.perf_counter()
    scored: list[tuple[float, Solution]] = []
    evaluated = 0
    for _, subset in candidates:
        if refine_continuous:
            idx = np.asarray(subset)
            sub_problem = PortfolioProblem(
                mu=problem.mu[idx],
                sigma=problem.sigma[np.ix_(idx, idx)],
                asset_names=tuple(problem.asset_names[i] for i in subset),
                w_min=problem.w_min,
                w_max=problem.w_max,
                budget=problem.budget,
                risk_aversion=problem.risk_aversion,
            )
            try:
                sub_sol = solve_continuous_mvo(sub_problem)
            except RuntimeError:
                continue
            weights = np.zeros(problem.n, dtype=float)
            weights[idx] = sub_sol.weights
        else:
            weights = np.zeros(problem.n, dtype=float)
            weights[list(subset)] = 1.0 / K
        obj = problem.objective_value(weights)
        if prev_weights is not None and turnover_cost_bps > 0:
            turnover = 0.5 * float(np.sum(np.abs(weights - prev_weights)))
            cost = turnover_cost_bps * turnover / 10000.0
            ranking_obj = obj + cost
        else:
            ranking_obj = obj
        evaluated += 1
        sol_candidate = Solution(
            weights=weights,
            objective=obj,
            feasible=_check_feasible(weights, problem),
            wall_time_s=0.0,
            solver=(
                f"tabu_selection+cvxpy[{num_reads}r/tenure={tenure}]"
                if refine_continuous
                else f"tabu_selection[{num_reads}r/tenure={tenure}]"
            ),
            solver_meta={"subset": list(subset)},
        )
        scored.append((ranking_obj, sol_candidate))
    refine_wall = time.perf_counter() - refine_t0

    best: Solution | None = None
    if scored:
        scored.sort(key=lambda pair: pair[0])
        best = scored[0][1]

    if best is None:
        first = sampleset.first
        sample_dict = {var: int(first.sample[var]) for var in sampleset.variables}
        subset_any = selection_bits_to_subset(sample_dict, problem.n)
        weights = np.zeros(problem.n, dtype=float)
        if subset_any:
            weights[list(subset_any)] = 1.0 / len(subset_any)
        return Solution(
            weights=weights,
            objective=problem.objective_value(weights),
            feasible=False,
            wall_time_s=sample_wall + refine_wall,
            solver=f"tabu_selection[{num_reads}r/tenure={tenure}]",
            solver_meta={
                "n_candidates": 0,
                "fallback_subset": list(subset_any),
                "sample_wall_s": sample_wall,
                "refine_wall_s": refine_wall,
            },
        )

    best.wall_time_s = sample_wall + refine_wall
    best.solver_meta.update({
        "sample_wall_s": sample_wall,
        "refine_wall_s": refine_wall,
        "n_candidates": len(candidates),
        "n_evaluated": evaluated,
        "num_reads": num_reads,
        "tenure": tenure,
    })
    return best


def _check_feasible(weights: np.ndarray, problem: PortfolioProblem, tol: float = 1e-6) -> bool:
    if abs(weights.sum() - problem.budget) > 1e-3:
        return False
    if np.any(weights < problem.w_min - tol):
        return False
    if np.any(weights > problem.w_max + tol):
        return False
    if problem.cardinality is not None:
        n_held = int((weights > 1e-6).sum())
        if n_held > problem.cardinality:
            return False
    return True
