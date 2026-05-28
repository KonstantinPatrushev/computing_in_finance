"""Experiment D: walk-forward business backtest.

Compares five portfolio strategies on two universes (S&P 500 and MOEX) via
rolling walk-forward:

1. Equal-weight 1/N — naive baseline
2. Continuous MVO (cvxpy) — no cardinality, continuous weights
3. Discrete MVO via MIQP (SCIP) — full classical pipeline
4. Discrete MVO via neal SA — quantum-inspired pipeline
5. Continuous MVO with post-hoc rounding to cardinality — practical
   classical heuristic used by many practitioners without a MIQP solver

Outputs JSONL rows (one per fold) plus a summary table under
``results/experiment_d_<universe>.csv`` and equity curves under
``results/final/figures/exp_d_<universe>_equity.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cif.backtest.walkforward import BacktestResult, run_walkforward
from cif.classical.continuous import solve_continuous_mvo
from cif.classical.milp import solve_miqp_scip
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_neal_selection, solve_with_tabu_selection

logger = logging.getLogger("cif.experiment_d")


def strategy_equal_weight(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
    return np.full(problem.n, 1.0 / problem.n)


def strategy_continuous(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
    # Drop cardinality for the continuous strategy
    p_no_card = PortfolioProblem(
        mu=problem.mu,
        sigma=problem.sigma,
        asset_names=problem.asset_names,
        w_min=problem.w_min,
        w_max=problem.w_max,
        budget=problem.budget,
        risk_aversion=problem.risk_aversion,
    )
    return solve_continuous_mvo(p_no_card).weights


def strategy_rounded(cardinality: int):
    def inner(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
        p_no_card = PortfolioProblem(
            mu=problem.mu,
            sigma=problem.sigma,
            asset_names=problem.asset_names,
            w_min=problem.w_min,
            w_max=problem.w_max,
            budget=problem.budget,
            risk_aversion=problem.risk_aversion,
        )
        cont = solve_continuous_mvo(p_no_card).weights
        top_idx = np.argsort(-cont)[:cardinality]
        refined = PortfolioProblem(
            mu=problem.mu[top_idx],
            sigma=problem.sigma[np.ix_(top_idx, top_idx)],
            asset_names=tuple(problem.asset_names[i] for i in top_idx),
            w_min=problem.w_min,
            w_max=problem.w_max,
            budget=problem.budget,
            risk_aversion=problem.risk_aversion,
        )
        sub_weights = solve_continuous_mvo(refined).weights
        w = np.zeros(problem.n, dtype=float)
        w[top_idx] = sub_weights
        return w
    return inner


def strategy_scip(cardinality: int, time_limit_s: float = 60.0):
    def inner(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
        sol = solve_miqp_scip(problem, cardinality=cardinality, time_limit_s=time_limit_s)
        if sol.feasible:
            return sol.weights
        return strategy_equal_weight(problem, fold_idx)
    return inner


def strategy_neal(
    cardinality: int,
    num_reads: int = 500,
    num_sweeps: int = 300,
    seed: int = 42,
    transaction_cost_bps: float = 10.0,
    persistent: bool = True,
):
    """Factory for the neal walk-forward strategy.

    When ``persistent=True`` each fold:
      * Seeds the candidate list with the previous fold's subset so the
        previous portfolio is always considered alongside SA's fresh picks.
      * Ranks candidates by ``objective + transaction_cost_bps·turnover``,
        so SA's new subset only wins if its return edge beats the rebalance
        cost it implies.
    """
    state = {"prev_weights": None, "prev_subset": None}

    def inner(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
        obj_scale = float(np.abs(problem.mu).max()) / cardinality + problem.risk_aversion * float(
            np.abs(problem.sigma).max()
        ) / (cardinality * cardinality)
        cp = 100.0 * max(obj_scale, 1e-6)
        sol = solve_with_neal_selection(
            problem,
            cardinality=cardinality,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed + fold_idx,
            cardinality_penalty=cp,
            top_subsets=50,
            warm_start_subset=state["prev_subset"] if persistent else None,
            prev_weights=state["prev_weights"] if persistent else None,
            turnover_cost_bps=transaction_cost_bps if persistent else 0.0,
        )
        state["prev_weights"] = sol.weights.copy()
        state["prev_subset"] = list(sol.solver_meta.get("subset", []))
        return sol.weights
    return inner


def strategy_tabu(
    cardinality: int,
    num_reads: int = 200,
    tenure: int = 20,
    seed: int = 42,
    transaction_cost_bps: float = 10.0,
    persistent: bool = True,
):
    """Tabu-based subset selection with continuous refinement.

    Mirrors :func:`strategy_neal` but swaps the Tabu sampler in. Tabu closes
    the 11% optimisation gap that neal SA leaves on dense Markowitz QUBOs,
    so this strategy is expected to match SCIP MIQP in objective at a
    fraction of its wall time.
    """
    state = {"prev_weights": None, "prev_subset": None}

    def inner(problem: PortfolioProblem, fold_idx: int) -> np.ndarray:
        obj_scale = float(np.abs(problem.mu).max()) / cardinality + problem.risk_aversion * float(
            np.abs(problem.sigma).max()
        ) / (cardinality * cardinality)
        cp = 100.0 * max(obj_scale, 1e-6)
        sol = solve_with_tabu_selection(
            problem,
            cardinality=cardinality,
            num_reads=num_reads,
            tenure=tenure,
            seed=seed + fold_idx,
            cardinality_penalty=cp,
            top_subsets=50,
            warm_start_subset=state["prev_subset"] if persistent else None,
            prev_weights=state["prev_weights"] if persistent else None,
            turnover_cost_bps=transaction_cost_bps if persistent else 0.0,
        )
        state["prev_weights"] = sol.weights.copy()
        state["prev_subset"] = list(sol.solver_meta.get("subset", []))
        return sol.weights
    return inner


def build_strategies(cardinality: int, transaction_cost_bps: float = 10.0) -> dict:
    return {
        "equal_weight": strategy_equal_weight,
        "continuous_mvo": strategy_continuous,
        "rounded_topk": strategy_rounded(cardinality),
        "scip_miqp": strategy_scip(cardinality),
        "neal_sa": strategy_neal(cardinality, transaction_cost_bps=transaction_cost_bps),
        "tabu_sa": strategy_tabu(cardinality, transaction_cost_bps=transaction_cost_bps),
    }


def run_for_universe(
    universe: str,
    cardinality: int,
    train_days: int,
    test_days: int,
    tc_bps: float,
    risk_aversion: float,
    out_dir: Path,
) -> dict[str, BacktestResult]:
    prices = pd.read_parquet(Path("data/processed") / f"{universe}_prices.parquet")
    logger.info(
        "universe=%s prices shape=%s range=%s..%s",
        universe,
        prices.shape,
        prices.index[0].date(),
        prices.index[-1].date(),
    )

    strategies = build_strategies(cardinality=cardinality, transaction_cost_bps=tc_bps)
    results: dict[str, BacktestResult] = {}
    for name, fn in strategies.items():
        logger.info("[%s/%s] backtest starting", universe, name)
        t0 = time.perf_counter()
        res = run_walkforward(
            prices=prices,
            strategy=fn,
            strategy_name=name,
            universe=universe,
            train_days=train_days,
            test_days=test_days,
            cardinality=cardinality,
            risk_aversion=risk_aversion,
            transaction_cost_bps=tc_bps,
            progress=True,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s/%s] done in %.1fs  folds=%d CAGR=%.3f Sharpe=%.3f turnover=%.3f",
            universe,
            name,
            elapsed,
            res.summary.get("n_folds", 0),
            res.summary.get("cagr", 0),
            res.summary.get("sharpe", 0),
            res.summary.get("annual_turnover", 0),
        )
        results[name] = res

    rows = []
    for name, res in results.items():
        row = {"strategy": name, "universe": universe, **res.summary}
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"experiment_d_{universe}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("wrote %s", summary_path)

    jsonl_path = out_dir / f"experiment_d_{universe}_folds.jsonl"
    with jsonl_path.open("w") as f:
        for name, res in results.items():
            for fold in res.folds:
                f.write(
                    json.dumps(
                        {
                            "strategy": name,
                            "universe": universe,
                            "fold": fold.fold,
                            "test_start": str(fold.test_start.date()),
                            "test_end": str(fold.test_end.date()),
                            "realised_return": fold.realised_return,
                            "turnover": fold.turnover,
                            "transaction_cost": fold.transaction_cost,
                            "solve_time_s": fold.solve_time_s,
                        }
                    )
                    + "\n"
                )
    logger.info("wrote %s", jsonl_path)

    return results


def plot_equity_curves(results: dict[str, BacktestResult], universe: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "equal_weight": "tab:gray",
        "continuous_mvo": "tab:blue",
        "rounded_topk": "tab:purple",
        "scip_miqp": "tab:orange",
        "neal_sa": "tab:green",
        "tabu_sa": "tab:red",
    }
    for name, res in results.items():
        if res.equity_curve is None:
            continue
        ax.plot(
            res.equity_curve.index,
            res.equity_curve.values,
            label=name,
            color=colors.get(name),
            linewidth=1.5,
        )
    ax.set_title(f"Experiment D: walk-forward equity curves — {universe.upper()}")
    ax.set_ylabel("Equity (initial = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / f"exp_d_{universe}_equity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", nargs="+", default=["sp500", "moex"])
    parser.add_argument("--cardinality-sp500", type=int, default=20)
    parser.add_argument("--cardinality-moex", type=int, default=10)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--tc-bps-sp500", type=float, default=10.0)
    parser.add_argument("--tc-bps-moex", type=float, default=30.0)
    parser.add_argument("--risk-aversion", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/final/figures"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    per_universe_settings = {
        "sp500": (args.cardinality_sp500, args.tc_bps_sp500),
        "moex": (args.cardinality_moex, args.tc_bps_moex),
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, BacktestResult]] = {}
    for universe in args.universes:
        K, bps = per_universe_settings[universe]
        results = run_for_universe(
            universe=universe,
            cardinality=K,
            train_days=args.train_days,
            test_days=args.test_days,
            tc_bps=bps,
            risk_aversion=args.risk_aversion,
            out_dir=args.out_dir,
        )
        plot_equity_curves(results, universe=universe, out_dir=args.figures_dir)
        all_results[universe] = results

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
