"""Experiment B: quality at a fixed wall-clock time budget.

For each ``(N, seed, time_budget)`` triple, every solver is given exactly the
same wall clock budget and must return its best feasible solution. We then
compare approximation ratio (vs the best solution any solver produced, or
SCIP with unlimited time if available).

This is the experiment that most directly answers the question "at time T,
who gives me the closer-to-optimal portfolio?" — the primary business
question for practitioners planning real-time rebalancing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from cif.classical.milp import solve_miqp_ecos_bb, solve_miqp_scip
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_neal_selection, solve_with_tabu_selection

logger = logging.getLogger("cif.experiment_b")


def make_problem(n: int, seed: int, K: int, risk_aversion: float = 2.0) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"B{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def run_neal_with_budget(problem: PortfolioProblem, budget_s: float, seed: int) -> dict:
    """Scale neal num_reads to approximately match a target wall time."""
    obj_scale = float(np.abs(problem.mu).max()) / problem.cardinality + problem.risk_aversion * float(
        np.abs(problem.sigma).max()
    ) / (problem.cardinality * problem.cardinality)
    cp = 100.0 * max(obj_scale, 1e-6)

    # Calibration run to estimate per-read wall time
    t0 = time.perf_counter()
    calib = solve_with_neal_selection(
        problem,
        num_reads=100,
        num_sweeps=300,
        seed=seed,
        top_subsets=25,
        cardinality_penalty=cp,
    )
    calib_wall = time.perf_counter() - t0

    if calib_wall >= budget_s * 0.9:
        return {
            "objective": calib.objective,
            "feasible": calib.feasible,
            "wall_time_s": calib_wall,
            "num_reads": 100,
            "num_sweeps": 300,
        }

    remaining = budget_s - calib_wall
    per_read = calib_wall / 100
    extra_reads = max(100, int(remaining / per_read * 0.9))

    t0 = time.perf_counter()
    full = solve_with_neal_selection(
        problem,
        num_reads=extra_reads,
        num_sweeps=300,
        seed=seed + 1,
        top_subsets=min(extra_reads // 2, 500),
        cardinality_penalty=cp,
    )
    full_wall = time.perf_counter() - t0

    better = calib if calib.objective < full.objective else full
    return {
        "objective": better.objective,
        "feasible": better.feasible,
        "wall_time_s": calib_wall + full_wall,
        "num_reads": 100 + extra_reads,
        "num_sweeps": 300,
    }


def run_tabu_with_budget(problem: PortfolioProblem, budget_s: float, seed: int) -> dict:
    """Scale Tabu ``num_reads`` to fit a target wall budget.

    Tabu's per-read cost is roughly constant for fixed (N, tenure), so we
    do one calibration run at ``num_reads=20`` to estimate it and then
    request enough additional reads to fill the budget. For budgets below
    the calibration cost we still report the calibration result (Tabu has
    no smaller granular unit than one read).
    """
    obj_scale = float(np.abs(problem.mu).max()) / problem.cardinality + problem.risk_aversion * float(
        np.abs(problem.sigma).max()
    ) / (problem.cardinality * problem.cardinality)
    cp = 100.0 * max(obj_scale, 1e-6)

    t0 = time.perf_counter()
    calib = solve_with_tabu_selection(
        problem,
        num_reads=20,
        tenure=20,
        seed=seed,
        cardinality_penalty=cp,
        top_subsets=10,
    )
    calib_wall = time.perf_counter() - t0

    if calib_wall >= budget_s * 0.9:
        return {
            "objective": calib.objective,
            "feasible": calib.feasible,
            "wall_time_s": calib_wall,
            "num_reads": 20,
            "tenure": 20,
        }

    remaining = budget_s - calib_wall
    per_read = max(calib_wall / 20, 1e-4)
    extra_reads = max(20, int(remaining / per_read * 0.8))
    # Hard cap so we don't burn forever
    extra_reads = min(extra_reads, 2000)

    t0 = time.perf_counter()
    full = solve_with_tabu_selection(
        problem,
        num_reads=extra_reads,
        tenure=20,
        seed=seed + 1,
        cardinality_penalty=cp,
        top_subsets=min(extra_reads // 2, 100),
    )
    full_wall = time.perf_counter() - t0

    better = calib if calib.objective < full.objective else full
    return {
        "objective": better.objective,
        "feasible": better.feasible,
        "wall_time_s": calib_wall + full_wall,
        "num_reads": 20 + extra_reads,
        "tenure": 20,
    }


def run_solvers(problem: PortfolioProblem, budget_s: float, seed: int) -> dict:
    out = {}

    # SCIP with time limit
    t0 = time.perf_counter()
    try:
        scip = solve_miqp_scip(problem, time_limit_s=budget_s)
        out["scip"] = {
            "objective": scip.objective,
            "feasible": scip.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "status": scip.solver_meta.get("status"),
        }
    except Exception as exc:
        out["scip"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # ECOS_BB
    t0 = time.perf_counter()
    try:
        ecos = solve_miqp_ecos_bb(problem, time_limit_s=budget_s)
        out["ecos_bb"] = {
            "objective": ecos.objective,
            "feasible": ecos.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "status": ecos.solver_meta.get("status"),
        }
    except Exception as exc:
        out["ecos_bb"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # neal scaled to the budget
    t0 = time.perf_counter()
    try:
        out["neal"] = run_neal_with_budget(problem, budget_s, seed)
    except Exception as exc:
        out["neal"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # tabu scaled to the budget
    t0 = time.perf_counter()
    try:
        out["tabu"] = run_tabu_with_budget(problem, budget_s, seed)
    except Exception as exc:
        out["tabu"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", nargs="+", type=int, default=[50, 100, 150, 200])
    parser.add_argument("--K-ratio", type=float, default=0.25)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.5, 1.0, 3.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--out", type=Path, default=Path("results/experiment_b.jsonl"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for n in args.N:
        K = max(3, int(round(args.K_ratio * n)))
        for seed in args.seeds:
            problem = make_problem(n=n, seed=seed, K=K)
            for budget in args.budgets:
                logger.info("N=%d K=%d seed=%d budget=%.1fs", n, K, seed, budget)
                results = run_solvers(problem, budget_s=budget, seed=seed)
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "N": n,
                    "K": K,
                    "seed": seed,
                    "budget_s": budget,
                    "results": results,
                }
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                for name, payload in results.items():
                    if "objective" in payload:
                        logger.info(
                            "  %-8s obj=%+.5f wall=%.2fs feasible=%s",
                            name,
                            payload["objective"],
                            payload["wall_time_s"],
                            payload.get("feasible", "?"),
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
