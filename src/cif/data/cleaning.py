"""Light-touch cleaning for daily price panels.

The policy is intentionally simple so it can be reasoned about in the report:

1. Forward-fill gaps up to ``max_ffill_days`` calendar days. Longer gaps remain
   NaN and are handled per-ticker.
2. Drop any ticker whose post-ffill NaN ratio exceeds ``max_nan_ratio``. This
   removes thinly traded names and tickers that started trading well after the
   window opens.
3. Trim the panel to the longest contiguous suffix where every surviving
   ticker has a price (no leading NaN). This guarantees the µ/Σ estimators
   downstream operate on a rectangular matrix.

The function also returns a small report describing what was dropped and why,
so the cleaning pass leaves an audit trail for the reproducibility document.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleaningReport:
    n_input_tickers: int
    n_output_tickers: int
    dropped_tickers: dict[str, str] = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    n_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "n_input_tickers": self.n_input_tickers,
            "n_output_tickers": self.n_output_tickers,
            "n_dropped": len(self.dropped_tickers),
            "dropped_tickers": self.dropped_tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_rows": self.n_rows,
        }


def clean_prices(
    prices: pd.DataFrame,
    max_ffill_days: int = 5,
    max_nan_ratio: float = 0.10,
    max_abs_daily_return: float = 0.60,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a wide price panel and return ``(cleaned, report)``.

    Corporate-action artefacts: when a provider fails to apply a
    stock-split adjustment (notably VTBR on MOEX ISS which had a 5000:1
    reverse split in 2024), the resulting daily return is many orders of
    magnitude larger than any plausible market move. We detect returns
    with ``|r| > max_abs_daily_return`` and rebase the price *level* from
    that day forward by dividing by ``(1 + r)``. The remaining panel then
    reflects the post-split price scale consistently, and the spurious
    one-day return is eliminated. This is applied *after* forward-fill
    and before the NaN-ratio filter.
    """
    report = CleaningReport(n_input_tickers=prices.shape[1], n_output_tickers=0)

    if prices.empty:
        return prices, report

    filled = prices.ffill(limit=max_ffill_days)

    # Detect and neutralise split-like jumps
    returns = filled.pct_change(fill_method=None)
    flagged = returns.abs() > max_abs_daily_return
    adjustments_made = 0
    if flagged.any().any():
        corrected = filled.copy()
        for ticker in flagged.columns:
            flagged_dates = filled.index[flagged[ticker].fillna(False)]
            for bad_date in flagged_dates:
                r = returns.loc[bad_date, ticker]
                if pd.isna(r):
                    continue
                factor = 1.0 + float(r)
                if factor == 0.0:
                    continue
                corrected.loc[bad_date:, ticker] = corrected.loc[bad_date:, ticker] / factor
                adjustments_made += 1
        filled = corrected
    report.dropped_tickers["_split_adjustments"] = f"{adjustments_made} price-level rebases"

    nan_ratio = filled.isna().mean()
    too_sparse = nan_ratio[nan_ratio > max_nan_ratio].index
    for ticker in too_sparse:
        report.dropped_tickers[str(ticker)] = (
            f"nan_ratio={nan_ratio[ticker]:.3f} > {max_nan_ratio}"
        )
    surviving = filled.drop(columns=too_sparse)

    if surviving.empty:
        report.n_output_tickers = 0
        return surviving, report

    first_valid_per_ticker = surviving.apply(lambda s: s.first_valid_index())
    panel_start = first_valid_per_ticker.max()
    if pd.isna(panel_start):
        return surviving.iloc[0:0], report
    rectangular = surviving.loc[panel_start:].dropna(how="any")

    report.n_output_tickers = rectangular.shape[1]
    report.n_rows = rectangular.shape[0]
    if not rectangular.empty:
        report.start_date = str(rectangular.index[0].date())
        report.end_date = str(rectangular.index[-1].date())

    return rectangular, report
