"""Experiment A: scalability of classical MIQP vs quantum-inspired SA.

Sweeps the universe size N over a representative range, fixes the cardinality
ratio K/N, and compares wall time and approximation quality across:

* SCIP MIQP (open-source production-grade)
* ECOS_BB MIQP (cvxpy bundled, vanilla branch-and-bound)
* DWave neal selection + cvxpy refinement (quantum-inspired SA pipeline)

Each (N, seed) pair is run independently; results are appended as JSONL rows
to ``results/experiment_a.jsonl``. Plots and aggregate tables are produced
by ``scripts/aggregate_experiment_a.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cif.classical.milp import solve_miqp_ecos_bb, solve_miqp_scip
from cif.data.bootstrap_synthetic import bootstrap_realistic_instance, load_real_covariance
from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_neal_selection, solve_with_tabu_selection

logger = logging.getLogger("cif.experiment_a")


def make_synthetic_problem(n: int, seed: int, K: int, risk_aversion: float = 2.0) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def make_real_problem(n: int, key: str, K: int, risk_aversion: float = 2.0,
                       train_start: str = "2015-01-01", train_end: str = "2024-12-31") -> PortfolioProblem:
    prices = pd.read_parquet(Path("data/processed") / f"{key}_prices.parquet")
    train = prices.loc[train_start:train_end]
    rets = log_returns(train)
    mu = annualised_mu(rets).values[:n]
    sigma = annualised_sigma(rets, method="ledoit_wolf").values[:n, :n]
    names = tuple(rets.columns[:n])
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=names,
        risk_aversion=risk_aversion,
        cardinality=K,
    )


_BOOTSTRAP_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def make_bootstrap_problem(
    n: int,
    seed: int,
    K: int,
    base_universe: str = "sp500",
    risk_aversion: float = 2.0,
    train_start: str = "2015-01-01",
    train_end: str = "2024-12-31",
) -> PortfolioProblem:
    """Make a problem with bootstrap-realistic Σ derived from a real universe."""
    cache_key = f"{base_universe}:{train_start}:{train_end}"
    if cache_key not in _BOOTSTRAP_CACHE:
        prices_path = Path("data/processed") / f"{base_universe}_prices.parquet"
        mu_real, sigma_real, _ = load_real_covariance(prices_path, train_start, train_end)
        _BOOTSTRAP_CACHE[cache_key] = (mu_real, sigma_real)
    mu_real, sigma_real = _BOOTSTRAP_CACHE[cache_key]
    mu, sigma = bootstrap_realistic_instance(mu_real, sigma_real, n_target=n, seed=seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"B{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def run_solvers(problem: PortfolioProblem, time_limit_s: float, neal_params: dict) -> dict:
    """Run all three solvers and return their results as a dict."""
    out = {}

    # SCIP
    t0 = time.perf_counter()
    try:
        scip = solve_miqp_scip(problem, time_limit_s=time_limit_s)
        out["scip"] = {
            "objective": scip.objective,
            "feasible": scip.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "n_nonzero": scip.solver_meta.get("n_nonzero", -1),
            "status": scip.solver_meta.get("status", "?"),
        }
    except Exception as exc:
        out["scip"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # ECOS_BB
    t0 = time.perf_counter()
    try:
        ecos = solve_miqp_ecos_bb(problem, time_limit_s=time_limit_s)
        out["ecos_bb"] = {
            "objective": ecos.objective,
            "feasible": ecos.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "n_nonzero": ecos.solver_meta.get("n_nonzero", -1),
            "status": ecos.solver_meta.get("status", "?"),
        }
    except Exception as exc:
        out["ecos_bb"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # neal SA
    t0 = time.perf_counter()
    try:
        neal_sol = solve_with_neal_selection(problem, **neal_params)
        out["neal"] = {
            "objective": neal_sol.objective,
            "feasible": neal_sol.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "subset": neal_sol.solver_meta.get("subset", []),
            "n_candidates": neal_sol.solver_meta.get("n_candidates", -1),
            "sa_wall_s": neal_sol.solver_meta.get("sa_wall_s", -1),
            "refine_wall_s": neal_sol.solver_meta.get("refine_wall_s", -1),
        }
    except Exception as exc:
        out["neal"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    # Tabu (calibrated default: 200 reads, tenure 20)
    t0 = time.perf_counter()
    try:
        tabu_sol = solve_with_tabu_selection(
            problem,
            num_reads=200,
            tenure=20,
            seed=neal_params.get("seed"),
            top_subsets=50,
            cardinality_penalty=neal_params.get("cardinality_penalty"),
        )
        out["tabu"] = {
            "objective": tabu_sol.objective,
            "feasible": tabu_sol.feasible,
            "wall_time_s": time.perf_counter() - t0,
            "subset": tabu_sol.solver_meta.get("subset", []),
            "n_candidates": tabu_sol.solver_meta.get("n_candidates", -1),
            "sample_wall_s": tabu_sol.solver_meta.get("sample_wall_s", -1),
            "refine_wall_s": tabu_sol.solver_meta.get("refine_wall_s", -1),
        }
    except Exception as exc:
        out["tabu"] = {"error": str(exc), "wall_time_s": time.perf_counter() - t0}

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", nargs="+", type=int, default=[20, 30, 50, 75, 100, 150, 200])
    parser.add_argument("--K-ratio", type=float, default=0.25, help="cardinality / N")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 2024, 31337])
    parser.add_argument("--time-limit", type=float, default=300.0, help="solver time limit in seconds")
    parser.add_argument(
        "--source",
        choices=["synthetic", "sp500", "moex", "bootstrap-sp500", "bootstrap-moex"],
        default="synthetic",
    )
    parser.add_argument("--out", type=Path, default=Path("results/experiment_a.jsonl"))
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=300)
    parser.add_argument("--top-subsets", type=int, default=100)
    parser.add_argument("--cardinality-penalty-mult", type=float, default=100.0)
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
            if args.source == "synthetic":
                problem = make_synthetic_problem(n=n, seed=seed, K=K)
            elif args.source.startswith("bootstrap-"):
                base = args.source.split("-", 1)[1]
                problem = make_bootstrap_problem(n=n, seed=seed, K=K, base_universe=base)
            else:
                problem = make_real_problem(n=n, key=args.source, K=K)
                if seed != args.seeds[0]:
                    continue  # real data is deterministic per N, no seed dependence

            obj_scale = float(np.abs(problem.mu).max()) / K + problem.risk_aversion * float(
                np.abs(problem.sigma).max()
            ) / (K * K)
            cp = args.cardinality_penalty_mult * max(obj_scale, 1e-6)

            neal_params = {
                "num_reads": args.num_reads,
                "num_sweeps": args.num_sweeps,
                "seed": seed,
                "top_subsets": args.top_subsets,
                "cardinality_penalty": cp,
            }

            logger.info("running N=%d K=%d seed=%d source=%s", n, K, seed, args.source)
            results = run_solvers(problem, time_limit_s=args.time_limit, neal_params=neal_params)

            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "N": n,
                "K": K,
                "K_ratio": args.K_ratio,
                "seed": seed,
                "source": args.source,
                "time_limit_s": args.time_limit,
                "neal_params": neal_params,
                "results": results,
            }
            with args.out.open("a") as f:
                f.write(json.dumps(row) + "\n")

            for solver_name, payload in results.items():
                if "objective" in payload:
                    logger.info(
                        "  %-8s obj=%+.5f wall=%.2fs feasible=%s",
                        solver_name,
                        payload["objective"],
                        payload["wall_time_s"],
                        payload.get("feasible", "?"),
                    )
                else:
                    logger.info("  %-8s ERROR %s", solver_name, payload.get("error"))

    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
