"""Aggregate Experiment D walk-forward results into business-facing artifacts.

Produces:

* ``results/final/experiment_d_business_table.csv`` — the central table with
  CAGR, Sharpe, Max DD, Annual Turnover, Annual Costs, Net Excess vs 1/N for
  every ``(strategy, universe)`` combination.
* ``results/final/figures/exp_d_<universe>_equity.png`` is produced by the
  runner; this script additionally generates
  ``exp_d_business_table.png`` — a heatmap view of the business table.
* ``results/final/experiment_d_business_summary.md`` — markdown summary for
  inclusion in the final report / defence slides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# AUM assumptions for dollar / rouble conversion
AUM_SP500 = 10_000_000  # USD
AUM_MOEX = 1_000_000_000  # RUB

STRATEGY_LABELS = {
    "equal_weight": "1/N Equal weight",
    "continuous_mvo": "Continuous MVO",
    "rounded_topk": "Continuous + Top-K rounding",
    "scip_miqp": "SCIP MIQP (discrete)",
    "neal_sa": "Neal SA (quantum-inspired)",
    "tabu_sa": "Tabu SA (quantum-inspired)",
}


def load_summaries(results_dir: Path, universes: list[str], suffix: str | None = None) -> pd.DataFrame:
    frames = []
    for u in universes:
        fname = f"experiment_d_{u}_summary.csv" if suffix is None else f"experiment_d_{u}_{suffix}_summary.csv"
        path = results_dir / fname
        if not path.exists():
            print(f"Skipping missing {path}")
            continue
        df = pd.read_csv(path)
        df["universe"] = u
        df["run_tag"] = suffix or "v1"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_business_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        universe = row["universe"]
        aum = AUM_SP500 if universe == "sp500" else AUM_MOEX
        currency = "$" if universe == "sp500" else "₽"
        annual_cost_frac = row["annual_costs_bps"] / 10000
        annual_cost_amount = annual_cost_frac * aum

        rows.append(
            {
                "Universe": universe.upper(),
                "Strategy": STRATEGY_LABELS.get(row["strategy"], row["strategy"]),
                "CAGR %": row["cagr"] * 100,
                "Sharpe": row["sharpe"],
                "Sortino": row["sortino"],
                "Max DD %": row["max_drawdown"] * 100,
                "Ann. Turnover %": row["annual_turnover"] * 100,
                "Ann. Costs bps": row["annual_costs_bps"],
                f"Ann. Costs ({currency} @ {aum:,})": annual_cost_amount,
                "Mean solve (s)": row.get("mean_solve_time_s", 0),
            }
        )
    table = pd.DataFrame(rows)
    return table


def compute_excess_vs_reference(table: pd.DataFrame, reference: str = "1/N Equal weight") -> pd.DataFrame:
    """Add a 'Net excess CAGR vs 1/N (bps)' column per universe."""
    out = table.copy()
    out["Net excess vs 1/N (bps)"] = 0.0
    for universe, sub in out.groupby("Universe"):
        ref_row = sub[sub["Strategy"] == reference]
        if ref_row.empty:
            continue
        ref_cagr = float(ref_row["CAGR %"].iloc[0])
        excess = (out.loc[out["Universe"] == universe, "CAGR %"] - ref_cagr) * 100  # bps
        out.loc[out["Universe"] == universe, "Net excess vs 1/N (bps)"] = excess
    return out


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = ["# Experiment D — business summary", "",
             "Walk-forward on S&P 500 (90 tickers, 2012-2025) and MOEX (28 tickers, 2014-2025).",
             "3-year training window, monthly rebalance, transaction costs 10 bps (SP) / 30 bps (MOEX).", ""]
    for universe in table["Universe"].unique():
        sub = table[table["Universe"] == universe].copy()
        lines.append(f"## {universe}")
        lines.append("")
        display_cols = [
            "Strategy", "CAGR %", "Sharpe", "Sortino", "Max DD %",
            "Ann. Turnover %", "Ann. Costs bps", "Net excess vs 1/N (bps)",
        ]
        lines.append(sub[display_cols].round(3).to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run tag (e.g. 'v2'); default reads v1 files")
    args = parser.parse_args()

    results_dir = Path("results")
    out_dir = Path("results/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(results_dir, universes=["sp500", "moex"], suffix=args.tag)
    if df.empty:
        print("No summaries to aggregate")
        return 1

    table = build_business_table(df)
    table = compute_excess_vs_reference(table)

    tag = args.tag or "v1"
    csv_path = out_dir / f"experiment_d_business_table_{tag}.csv"
    table.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    md_path = out_dir / f"experiment_d_business_summary_{tag}.md"
    write_markdown(table, md_path)

    pd.options.display.float_format = "{:.2f}".format
    display_cols = [
        "Universe", "Strategy", "CAGR %", "Sharpe", "Sortino",
        "Max DD %", "Ann. Turnover %", "Ann. Costs bps", "Net excess vs 1/N (bps)",
    ]
    print()
    print(table[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
