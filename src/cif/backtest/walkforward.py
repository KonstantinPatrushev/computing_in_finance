"""Walk-forward backtest engine for comparing portfolio strategies.

A single ``run_walkforward`` call rolls a ``(train, test)`` window through a
price panel, re-estimates ``(μ, Σ)`` each fold, calls a user-supplied
``strategy`` function to produce weights, marks the resulting portfolio to
market during the test window, accumulates returns, and tracks all
transaction costs triggered by weight changes between folds.

The engine is strategy-agnostic: a ``strategy`` is any callable that takes a
``PortfolioProblem`` plus the current fold index and returns a weight vector
of length ``N``. Classical (cvxpy, SCIP), quantum-inspired (``neal``), and
even random baselines can all plug in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.problem import PortfolioProblem


StrategyFn = Callable[[PortfolioProblem, int], np.ndarray]


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    weights: np.ndarray
    realised_return: float
    realised_std: float
    turnover: float
    transaction_cost: float
    solve_time_s: float


@dataclass
class BacktestResult:
    strategy_name: str
    universe: str
    folds: list[FoldResult] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    summary: dict = field(default_factory=dict)


def _make_problem(
    returns_train: pd.DataFrame,
    cardinality: int | None,
    risk_aversion: float,
    w_min: float,
    w_max: float,
) -> PortfolioProblem:
    mu = annualised_mu(returns_train).values
    sigma = annualised_sigma(returns_train, method="ledoit_wolf").values
    names = tuple(returns_train.columns)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=names,
        risk_aversion=risk_aversion,
        cardinality=cardinality,
        w_min=w_min,
        w_max=w_max,
    )


def run_walkforward(
    prices: pd.DataFrame,
    strategy: StrategyFn,
    strategy_name: str,
    universe: str,
    train_days: int = 756,  # 3 years of trading days
    test_days: int = 21,    # ~1 month
    step_days: int | None = None,
    cardinality: int | None = None,
    risk_aversion: float = 2.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1.0,
    progress: bool = False,
) -> BacktestResult:
    """Run one strategy through a rolling walk-forward.

    Parameters
    ----------
    prices:
        Wide price panel, date-indexed, one column per ticker. Gaps must
        already be cleaned (see :mod:`cif.data.cleaning`).
    strategy:
        ``fn(problem, fold_index) -> weights``. The caller closes over any
        solver-specific kwargs.
    train_days, test_days:
        Window sizes in trading days.
    step_days:
        How far to advance the window per fold. Default ``test_days``
        (contiguous, no overlap in test windows).
    transaction_cost_bps:
        Cost per unit of turnover in basis points. Turnover is
        ``sum(|Δw|) / 2`` — the conventional definition that charges each
        asset's share once per rebalance (half for selling, half for
        buying).
    initial_capital:
        Starting capital used for the equity curve.
    """
    if step_days is None:
        step_days = test_days

    dates = prices.index
    if len(dates) < train_days + test_days:
        raise ValueError(
            f"Need at least {train_days + test_days} days, got {len(dates)}"
        )

    n_assets = prices.shape[1]
    prev_weights = np.zeros(n_assets, dtype=float)
    equity = initial_capital
    equity_dates: list[pd.Timestamp] = []
    equity_values: list[float] = []

    folds: list[FoldResult] = []
    fold_idx = 0
    i = 0
    while i + train_days + test_days <= len(dates):
        train_slice = prices.iloc[i : i + train_days]
        test_slice = prices.iloc[i + train_days : i + train_days + test_days]

        train_returns = log_returns(train_slice)
        if train_returns.empty:
            break

        problem = _make_problem(
            returns_train=train_returns,
            cardinality=cardinality,
            risk_aversion=risk_aversion,
            w_min=w_min,
            w_max=w_max,
        )

        t0 = time.perf_counter()
        weights = strategy(problem, fold_idx)
        solve_time = time.perf_counter() - t0
        weights = np.asarray(weights, dtype=float)

        turnover = 0.5 * float(np.sum(np.abs(weights - prev_weights)))
        cost_frac = turnover * transaction_cost_bps / 10000.0

        # Realized simple return over the test window
        test_simple_returns = test_slice.pct_change().dropna()
        if test_simple_returns.empty:
            i += step_days
            fold_idx += 1
            continue
        daily_portfolio_returns = test_simple_returns.values @ weights
        fold_return = float(np.prod(1.0 + daily_portfolio_returns) - 1.0)
        fold_std = float(np.std(daily_portfolio_returns) * np.sqrt(252))
        net_fold_return = fold_return - cost_frac

        equity *= 1.0 + net_fold_return

        for d, r in zip(test_simple_returns.index, daily_portfolio_returns):
            equity_dates.append(d)
            equity_values.append(equity)

        folds.append(
            FoldResult(
                fold=fold_idx,
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                weights=weights,
                realised_return=fold_return,
                realised_std=fold_std,
                turnover=turnover,
                transaction_cost=cost_frac,
                solve_time_s=solve_time,
            )
        )

        prev_weights = weights
        fold_idx += 1
        i += step_days

        if progress and fold_idx % 5 == 0:
            print(
                f"  [{strategy_name}] fold {fold_idx}: "
                f"eq={equity:.3f}, turnover={turnover:.3f}, "
                f"solve={solve_time*1000:.0f}ms"
            )

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates)) if equity_values else None
    summary = _summarise(folds, equity_curve, transaction_cost_bps, initial_capital)

    return BacktestResult(
        strategy_name=strategy_name,
        universe=universe,
        folds=folds,
        equity_curve=equity_curve,
        summary=summary,
    )


def _summarise(
    folds: list[FoldResult],
    equity_curve: pd.Series | None,
    tc_bps: float,
    initial_capital: float,
) -> dict:
    if not folds or equity_curve is None or equity_curve.empty:
        return {}

    final_equity = float(equity_curve.iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = (end - start).days / 365.25
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    total_turnover = sum(f.turnover for f in folds)
    total_costs = sum(f.transaction_cost for f in folds)
    mean_solve = float(np.mean([f.solve_time_s for f in folds]))
    annual_turnover = total_turnover / years if years > 0 else total_turnover
    annual_costs_bps = total_costs / years * 10000 if years > 0 else total_costs * 10000

    return {
        "n_folds": len(folds),
        "years": years,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "annual_turnover": annual_turnover,
        "annual_costs_bps": annual_costs_bps,
        "total_turnover": total_turnover,
        "total_costs_frac": total_costs,
        "mean_solve_time_s": mean_solve,
        "tc_bps": tc_bps,
    }
