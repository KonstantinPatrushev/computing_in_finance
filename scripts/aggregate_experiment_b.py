"""Aggregate Experiment B (quality at fixed time budget) into a heatmap.

Produces:

* ``results/final/experiment_b_summary.csv`` — one row per (N, budget,
  solver) with median objective, median wall time, and median approximation
  ratio across seeds.
* ``results/final/figures/exp_b_quality_heatmap.png`` — a 2D heatmap of
  median approximation ratio for every solver across the (N, budget) grid.
* ``results/final/figures/exp_b_winner.png`` — at each (N, budget) cell, the
  solver that achieved the best (lowest) median objective. Colour-coded by
  winner. This is the "who should I use for budget T at universe N" plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOLVERS = ["scip", "ecos_bb", "neal", "tabu"]
COLORS = {"scip": "tab:blue", "ecos_bb": "tab:orange", "neal": "tab:green", "tabu": "tab:red"}
WINNER_COLORS = {"scip": "#1f77b4", "ecos_bb": "#ff7f0e", "neal": "#2ca02c", "tabu": "#d62728"}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            for solver, payload in r["results"].items():
                if "objective" not in payload:
                    continue
                rows.append(
                    {
                        "N": r["N"],
                        "K": r["K"],
                        "seed": r["seed"],
                        "budget_s": r["budget_s"],
                        "solver": solver,
                        "objective": payload["objective"],
                        "wall_time_s": payload["wall_time_s"],
                        "feasible": payload.get("feasible", False),
                    }
                )
    return pd.DataFrame(rows)


def add_approx_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """For every (N, seed, budget), compute gap to best objective across solvers."""
    df = df.copy()
    grp = df.groupby(["N", "seed", "budget_s"])["objective"]
    df["best_obj"] = grp.transform("min")
    df["gap_pct"] = (df["objective"] - df["best_obj"]) / np.abs(df["best_obj"]) * 100
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["N", "K", "budget_s", "solver"])
    return grp.agg(
        objective_median=("objective", "median"),
        gap_median_pct=("gap_pct", "median"),
        wall_time_median=("wall_time_s", "median"),
        feasible_rate=("feasible", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()


def plot_quality_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    Ns = sorted(summary["N"].unique())
    budgets = sorted(summary["budget_s"].unique())
    solvers = SOLVERS

    fig, axes = plt.subplots(1, len(solvers), figsize=(4 * len(solvers), 4), sharey=True)
    vmin = 0
    vmax = max(20, float(summary["gap_median_pct"].quantile(0.95)))
    for ax, solver in zip(axes, solvers):
        sub = summary[summary["solver"] == solver]
        mat = np.full((len(budgets), len(Ns)), np.nan)
        for i, b in enumerate(budgets):
            for j, n in enumerate(Ns):
                row = sub[(sub["budget_s"] == b) & (sub["N"] == n)]
                if not row.empty:
                    mat[i, j] = float(row["gap_median_pct"].iloc[0])
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(Ns)
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels([f"{b:g}s" for b in budgets])
        ax.set_xlabel("N")
        ax.set_title(solver)
        for i in range(len(budgets)):
            for j in range(len(Ns)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                            color="black" if v < vmax * 0.5 else "white")
    axes[0].set_ylabel("Time budget")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Median gap to best (%)", shrink=0.7)
    fig.suptitle("Experiment B: solution quality at fixed time budget")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_winner_grid(summary: pd.DataFrame, out_path: Path) -> None:
    """For each (N, budget) cell, highlight the solver with the best median objective."""
    Ns = sorted(summary["N"].unique())
    budgets = sorted(summary["budget_s"].unique())

    winner = np.full((len(budgets), len(Ns)), -1)
    winner_name = np.full((len(budgets), len(Ns)), "", dtype=object)
    for i, b in enumerate(budgets):
        for j, n in enumerate(Ns):
            sub = summary[(summary["budget_s"] == b) & (summary["N"] == n)]
            if sub.empty:
                continue
            best = sub.loc[sub["objective_median"].idxmin()]
            winner_name[i, j] = best["solver"]
            winner[i, j] = SOLVERS.index(best["solver"])

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.matplotlib.colors.ListedColormap([WINNER_COLORS[s] for s in SOLVERS])
    im = ax.imshow(winner, cmap=cmap, vmin=0, vmax=len(SOLVERS) - 1, aspect="auto")
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(budgets)))
    ax.set_yticklabels([f"{b:g}s" for b in budgets])
    ax.set_xlabel("Universe size N")
    ax.set_ylabel("Time budget")
    ax.set_title("Experiment B: best solver at each (N, budget) cell")
    for i in range(len(budgets)):
        for j in range(len(Ns)):
            ax.text(j, i, winner_name[i, j], ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=WINNER_COLORS[s]) for s in SOLVERS]
    ax.legend(handles, SOLVERS, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=Path, default=Path("results/experiment_b.jsonl"))
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

    summary.to_csv(args.out_dir / "experiment_b_summary.csv", index=False)
    plot_quality_heatmap(summary, figures_dir / "exp_b_quality_heatmap.png")
    plot_winner_grid(summary, figures_dir / "exp_b_winner.png")

    pd.options.display.float_format = "{:.3f}".format
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
