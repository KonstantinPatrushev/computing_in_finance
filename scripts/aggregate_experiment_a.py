"""Aggregate the JSONL output from ``experiment_a_scalability.py``.

Produces:

* ``results/final/experiment_a_summary.csv`` — one row per ``(N, K, source)``
  with median, mean, and 95% interval of wall time and approximation ratio
  for every solver.
* ``results/final/figures/exp_a_scalability.png`` — log-log wall-time vs N
  with all three solvers overlaid; the headline plot for Experiment A.
* ``results/final/figures/exp_a_quality.png`` — approximation ratio vs N for
  the two non-reference solvers (using SCIP as the reference where SCIP
  finished, otherwise the best objective seen).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            n = row["N"]
            K = row["K"]
            seed = row["seed"]
            source = row["source"]
            for solver, payload in row["results"].items():
                rows.append(
                    {
                        "N": n,
                        "K": K,
                        "seed": seed,
                        "source": source,
                        "solver": solver,
                        "objective": payload.get("objective", np.nan),
                        "wall_time_s": payload.get("wall_time_s", np.nan),
                        "feasible": payload.get("feasible", False),
                        "error": payload.get("error"),
                    }
                )
    return pd.DataFrame(rows)


def add_approx_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Per (N, K, seed) compute approximation ratio = obj / best_obj."""
    df = df.copy()
    df["best_obj"] = df.groupby(["N", "K", "seed", "source"])[
        "objective"
    ].transform(lambda s: s[df["feasible"][s.index]].min() if (df["feasible"][s.index]).any() else np.nan)
    df["approx_ratio"] = df["objective"] / df["best_obj"]
    df["gap_pct"] = (df["objective"] - df["best_obj"]) / np.abs(df["best_obj"]) * 100
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["source", "N", "K", "solver"])
    summary = grouped.agg(
        wall_time_median=("wall_time_s", "median"),
        wall_time_mean=("wall_time_s", "mean"),
        wall_time_std=("wall_time_s", "std"),
        objective_median=("objective", "median"),
        gap_median_pct=("gap_pct", "median"),
        gap_mean_pct=("gap_pct", "mean"),
        feasible_rate=("feasible", "mean"),
        n_runs=("seed", "count"),
    ).reset_index()
    return summary


def plot_scalability(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"scip": "tab:blue", "ecos_bb": "tab:orange", "neal": "tab:green", "tabu": "tab:red"}
    markers = {"scip": "o", "ecos_bb": "s", "neal": "^", "tabu": "D"}
    for solver, sub in summary.groupby("solver"):
        sub_sorted = sub.sort_values("N")
        ax.plot(
            sub_sorted["N"],
            sub_sorted["wall_time_median"],
            marker=markers.get(solver, "x"),
            label=solver,
            color=colors.get(solver),
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("Universe size N")
    ax.set_ylabel("Wall time (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("Experiment A: wall-time scalability of MIQP vs quantum-inspired SA")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"scip": "tab:blue", "ecos_bb": "tab:orange", "neal": "tab:green", "tabu": "tab:red"}
    markers = {"scip": "o", "ecos_bb": "s", "neal": "^", "tabu": "D"}
    for solver, sub in summary.groupby("solver"):
        sub_sorted = sub.sort_values("N")
        ax.plot(
            sub_sorted["N"],
            sub_sorted["gap_median_pct"],
            marker=markers.get(solver, "x"),
            label=solver,
            color=colors.get(solver),
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("Universe size N")
    ax.set_ylabel("Median gap to best (%)")
    ax.set_title("Experiment A: solution quality (gap to best feasible objective)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=Path, default=Path("results/experiment_a.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    df = load_jsonl(args.in_file)
    if df.empty:
        print(f"No rows in {args.in_file}")
        return 1
    df = add_approx_ratio(df)
    summary = summarise(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.out_dir / "experiment_a_summary.csv", index=False)
    plot_scalability(summary, figures_dir / "exp_a_scalability.png")
    plot_quality(summary, figures_dir / "exp_a_quality.png")

    pd.options.display.float_format = "{:.4f}".format
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
