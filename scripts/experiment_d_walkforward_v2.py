"""Experiment D v2: walk-forward with tight diversification constraints.

The v1 run showed that with ``w_max=1.0`` and ``risk_aversion=2.0`` the
continuous MVO already concentrates into ≤ ``K`` assets, so the cardinality
constraint never binds and discrete methods produce identical portfolios.
v2 tightens the problem:

* ``w_max = 0.10`` (hard cap at 10% per asset)
* ``K = 15`` (tight cardinality)
* ``risk_aversion = 5.0`` (more risk-averse baseline)

Under these constraints ``continuous_mvo`` **must** spread weight across at
least 10 assets (since max 10% × 10 = 100% budget), and the K=15 constraint
binds when ``continuous_mvo`` wants to hold > 15 names. This separates the
strategies cleanly.

Everything else is the same as :mod:`scripts.experiment_d_walkforward`.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd

from cif.backtest.walkforward import run_walkforward

# Load sibling module without requiring scripts/ to be a package
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_exp_d_v1", _HERE / "experiment_d_walkforward.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_exp_d_v1"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
build_strategies = _mod.build_strategies
plot_equity_curves = _mod.plot_equity_curves

logger = logging.getLogger("cif.experiment_d_v2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", nargs="+", default=["sp500", "moex"])
    parser.add_argument("--cardinality-sp500", type=int, default=15)
    parser.add_argument("--cardinality-moex", type=int, default=8)
    parser.add_argument("--w-max-sp500", type=float, default=0.10)
    parser.add_argument("--w-max-moex", type=float, default=0.20)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--tc-bps-sp500", type=float, default=10.0)
    parser.add_argument("--tc-bps-moex", type=float, default=30.0)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/final/figures"))
    parser.add_argument("--suffix", default="v2")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    per_universe = {
        "sp500": (args.cardinality_sp500, args.w_max_sp500, args.tc_bps_sp500),
        "moex": (args.cardinality_moex, args.w_max_moex, args.tc_bps_moex),
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for universe in args.universes:
        K, w_max, bps = per_universe[universe]
        prices = pd.read_parquet(Path("data/processed") / f"{universe}_prices.parquet")
        strategies = build_strategies(cardinality=K, transaction_cost_bps=bps)

        rows = []
        results_per_strategy = {}
        for name, fn in strategies.items():
            logger.info("[%s/%s/%s] running K=%d w_max=%.2f bps=%.1f", universe, args.suffix, name, K, w_max, bps)
            res = run_walkforward(
                prices=prices,
                strategy=fn,
                strategy_name=name,
                universe=universe,
                train_days=args.train_days,
                test_days=args.test_days,
                cardinality=K,
                risk_aversion=args.risk_aversion,
                w_min=0.0,
                w_max=w_max,
                transaction_cost_bps=bps,
                progress=True,
            )
            logger.info(
                "[%s/%s/%s] CAGR=%.3f Sharpe=%.3f turn=%.3f costs=%.1fbps",
                universe, args.suffix, name,
                res.summary.get("cagr", 0),
                res.summary.get("sharpe", 0),
                res.summary.get("annual_turnover", 0),
                res.summary.get("annual_costs_bps", 0),
            )
            rows.append({"strategy": name, "universe": universe, **res.summary})
            results_per_strategy[name] = res

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(args.out_dir / f"experiment_d_{universe}_{args.suffix}_summary.csv", index=False)
        plot_equity_curves(
            results_per_strategy,
            universe=f"{universe}_{args.suffix}",
            out_dir=args.figures_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
