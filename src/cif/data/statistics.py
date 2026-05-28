"""Return and moment estimators for portfolio optimisation.

All functions accept a wide price/return DataFrame (date index, ticker columns)
and return either a Series or a DataFrame with consistent dtypes. The
annualisation convention is ``periods_per_year=252`` (US trading days). MOEX
also uses ~252 trading days a year so the same default works for both
universes; pass an explicit value if you ever switch frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PERIODS_PER_YEAR = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from an adjusted-close panel."""
    if prices.empty:
        return prices
    return np.log(prices / prices.shift(1)).dropna(how="any")


def annualised_mu(
    returns: pd.DataFrame,
    periods_per_year: int = PERIODS_PER_YEAR,
    method: str = "mean",
    halflife: int | None = None,
) -> pd.Series:
    """Annualised expected-return vector.

    Parameters
    ----------
    method:
        ``"mean"`` (sample average) or ``"ema"`` (exponentially weighted mean).
    halflife:
        Required when ``method == "ema"``; ignored otherwise. Expressed in
        rows (i.e. trading days).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    if method == "mean":
        mu_daily = returns.mean(axis=0)
    elif method == "ema":
        if halflife is None:
            raise ValueError("halflife is required when method='ema'")
        mu_daily = returns.ewm(halflife=halflife, adjust=False).mean().iloc[-1]
    else:
        raise ValueError(f"Unknown mu method: {method!r}")
    return mu_daily * periods_per_year


def annualised_sigma(
    returns: pd.DataFrame,
    periods_per_year: int = PERIODS_PER_YEAR,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Annualised covariance matrix.

    Parameters
    ----------
    method:
        ``"sample"`` (plain ``np.cov`` style estimator) or ``"ledoit_wolf"``
        (shrinkage to identity scaled by average sample variance — the
        textbook default for large N).
    """
    if returns.empty:
        return pd.DataFrame()
    if method == "sample":
        cov_daily = returns.cov().to_numpy()
    elif method == "ledoit_wolf":
        estimator = LedoitWolf().fit(returns.to_numpy())
        cov_daily = estimator.covariance_
    else:
        raise ValueError(f"Unknown sigma method: {method!r}")

    cov_annual = cov_daily * periods_per_year
    return pd.DataFrame(cov_annual, index=returns.columns, columns=returns.columns)


def compute_moments(
    prices: pd.DataFrame,
    mu_method: str = "mean",
    sigma_method: str = "ledoit_wolf",
    halflife: int | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Pipeline helper: prices → (μ, Σ, returns).

    Returned in that order so the caller can persist returns separately for
    use in walk-forward backtests later.
    """
    rets = log_returns(prices)
    mu = annualised_mu(rets, periods_per_year=periods_per_year, method=mu_method, halflife=halflife)
    sigma = annualised_sigma(rets, periods_per_year=periods_per_year, method=sigma_method)
    return mu, sigma, rets
