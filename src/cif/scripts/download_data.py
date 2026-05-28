"""Download price panels for S&P 500 and/or MOEX and persist to parquet.

Usage::

    python -m cif.scripts.download_data --universe sp500 --top 100
    python -m cif.scripts.download_data --universe moex
    python -m cif.scripts.download_data --universe both

Outputs (relative to ``--root``)::

    data/raw/universe_<key>.json     # ticker snapshot
    data/raw/<key>_prices_raw.parquet
    data/processed/<key>_prices.parquet
    data/processed/<key>_returns.parquet
    data/processed/<key>_provenance.json

This script is **local-only** — the HPC notebook reads the parquet files that
this command produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from cif.data.cleaning import clean_prices
from cif.data.providers import fetch_moex_prices, fetch_yfinance_prices
from cif.data.statistics import log_returns
from cif.data.universe import resolve_universe

logger = logging.getLogger("cif.download")

DEFAULTS = {
    "sp500": {"start": "2011-01-01", "end": "2025-12-31", "top": 100},
    "moex": {"start": "2014-01-01", "end": "2025-12-31", "top": None},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def download_universe(
    key: str,
    root: Path,
    start: str,
    end: str,
    top: int | None,
    refresh_universe: bool,
) -> dict:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    snapshot = resolve_universe(key, raw_dir=raw_dir, refresh=refresh_universe, top_n=top)
    logger.info("universe %s: %d tickers (source=%s)", key, len(snapshot.tickers), snapshot.source)

    if key == "sp500":
        raw_prices = fetch_yfinance_prices(snapshot.tickers, start=start, end=end)
    elif key == "moex":
        raw_prices = fetch_moex_prices(snapshot.tickers, start=start, end=end)
    else:
        raise ValueError(f"Unknown universe key: {key!r}")

    raw_path = raw_dir / f"{key}_prices_raw.parquet"
    raw_prices.to_parquet(raw_path)
    logger.info("wrote raw prices to %s shape=%s", raw_path, raw_prices.shape)

    cleaned, report = clean_prices(raw_prices)
    processed_path = processed_dir / f"{key}_prices.parquet"
    cleaned.to_parquet(processed_path)
    logger.info(
        "wrote cleaned prices to %s shape=%s dropped=%d",
        processed_path, cleaned.shape, len(report.dropped_tickers),
    )

    returns = log_returns(cleaned)
    returns_path = processed_dir / f"{key}_returns.parquet"
    returns.to_parquet(returns_path)
    logger.info("wrote returns to %s shape=%s", returns_path, returns.shape)

    provenance = {
        "key": key,
        "downloaded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "start": start,
        "end": end,
        "universe_snapshot": snapshot.to_dict(),
        "raw_prices": {
            "path": str(raw_path.relative_to(root)),
            "shape": list(raw_prices.shape),
            "sha256": _sha256(raw_path),
        },
        "cleaned_prices": {
            "path": str(processed_path.relative_to(root)),
            "shape": list(cleaned.shape),
            "sha256": _sha256(processed_path),
        },
        "returns": {
            "path": str(returns_path.relative_to(root)),
            "shape": list(returns.shape),
            "sha256": _sha256(returns_path),
        },
        "cleaning_report": report.to_dict(),
    }

    provenance_path = processed_dir / f"{key}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    logger.info("wrote provenance to %s", provenance_path)

    return provenance


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download S&P 500 and/or MOEX price panels.")
    parser.add_argument("--universe", choices=["sp500", "moex", "both"], required=True)
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--start", type=str, default=None, help="ISO start date")
    parser.add_argument("--end", type=str, default=None, help="ISO end date (exclusive)")
    parser.add_argument("--top", type=int, default=None,
                        help="Cap the universe to the top-N tickers from the source order")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="Force re-fetch of the constituent list")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    keys = ["sp500", "moex"] if args.universe == "both" else [args.universe]
    for key in keys:
        defaults = DEFAULTS[key]
        download_universe(
            key=key,
            root=args.root,
            start=args.start or defaults["start"],
            end=args.end or defaults["end"],
            top=args.top if args.top is not None else defaults["top"],
            refresh_universe=args.refresh_universe,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
