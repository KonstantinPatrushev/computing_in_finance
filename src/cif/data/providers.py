"""Price data providers for S&P 500 (yfinance) and MOEX (apimoex).

Both providers return a wide DataFrame indexed by date with one column per
ticker holding the adjusted close. Missing tickers are silently dropped — the
caller decides what to do with the survivors via :mod:`cif.data.cleaning`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date

import apimoex
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_yfinance_prices(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
    batch_size: int = 50,
    sleep_between_batches: float = 1.0,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers:
        Iterable of Yahoo-style symbols (e.g. ``"BRK-B"``).
    start, end:
        ISO dates or ``date`` objects, inclusive of start and exclusive of end
        (yfinance convention).
    batch_size:
        Number of tickers to request per ``yf.download`` call. yfinance imposes
        practical limits, so we batch.
    sleep_between_batches:
        Wall sleep between batches in seconds. Cheap insurance against rate
        limiting.
    """
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info("yfinance batch %d/%d (%d tickers)", i // batch_size + 1,
                    (len(tickers) + batch_size - 1) // batch_size, len(batch))
        downloaded = yf.download(
            batch,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if downloaded is None or downloaded.empty:
            logger.warning("yfinance returned empty frame for batch starting %s", batch[0])
            continue

        if isinstance(downloaded.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker in downloaded.columns.get_level_values(0):
                    series = downloaded[ticker].get("Close")
                    if series is not None:
                        frames.append(series.rename(ticker).to_frame())
        else:
            close = downloaded.get("Close")
            if close is not None:
                frames.append(close.rename(batch[0]).to_frame())

        if i + batch_size < len(tickers):
            time.sleep(sleep_between_batches)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices


def fetch_moex_prices(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
    board: str = "TQBR",
    sleep_between_calls: float = 0.1,
) -> pd.DataFrame:
    """Download daily closes from MOEX ISS via :mod:`apimoex`.

    One HTTP call per ticker — there is no batch endpoint on the TQBR board.
    Failed tickers are skipped with a warning.
    """
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    series_list: list[pd.Series] = []
    with requests.Session() as session:
        for idx, ticker in enumerate(tickers, start=1):
            try:
                rows = apimoex.get_board_history(
                    session,
                    security=ticker,
                    start=str(start),
                    end=str(end),
                    board=board,
                    columns=("TRADEDATE", "CLOSE"),
                )
            except Exception as exc:
                logger.warning("MOEX fetch failed for %s: %s", ticker, exc)
                continue

            if not rows:
                logger.warning("MOEX returned empty rows for %s", ticker)
                continue

            df = pd.DataFrame(rows)
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
            df = df.set_index("TRADEDATE").rename(columns={"CLOSE": ticker})
            series_list.append(df[ticker])
            logger.info("MOEX %d/%d %s rows=%d", idx, len(tickers), ticker, len(df))
            time.sleep(sleep_between_calls)

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1).sort_index()
    prices.index.name = "date"
    return prices
