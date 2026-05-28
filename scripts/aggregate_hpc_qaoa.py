"""Aggregate HPC QAOA results (cHARISMa cluster run) into business artefacts.

Inputs
------
- ``results/hpc_qaoa.jsonl`` — one row per (N, p, seed) cell.

Outputs
-------
- ``results/final/experiment_c_hpc_summary.csv`` — wide summary table
  (median gap, median wall time per (N, p)).
- ``results/final/figures/exp_c_hpc_scaling.png`` — wall time vs N for
  p ∈ {1, 2}, log-y axis.
- ``results/final/figures/exp_c_hpc_quality.png`` — gap-to-BF vs N for
  p ∈ {1, 2}.
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
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["N", "K", "p"])
    return grp.agg(
        n_seeds=("seed", "count"),
        train_time_median=("qaoa_train_time_s", "median"),
        sample_time_median=("qaoa_sample_time_s", "median"),
        expected_obj_median=("expected_obj", "median"),
        p_valid_median=("p_valid", "median"),
        best_obj_median=("best_obj", "median"),
        gap_pct_median=("gap_to_bf_pct", "median"),
        gap_pct_mean=("gap_to_bf_pct", "mean"),
        gap_pct_std=("gap_to_bf_pct", "std"),
        bf_optimum=("bf_optimum", "first"),
    ).reset_index()


def plot_scaling(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, color, marker in [(1, "tab:blue", "o"), (2, "tab:red", "s")]:
        sub = summary[summary["p"] == p].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"], sub["train_time_median"], marker=marker, color=color,
                linewidth=2, markersize=8, label=f"p={p}")
    ax.set_xlabel("Universe size N (qubits)")
    ax.set_ylabel("Training wall time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("Experiment C-HPC: QAOA training time vs N on V100 (pure-numpy statevector)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_quality(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, color, marker in [(1, "tab:blue", "o"), (2, "tab:red", "s")]:
        sub = summary[summary["p"] == p].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"], sub["gap_pct_median"], marker=marker, color=color,
                linewidth=2, markersize=8, label=f"p={p}")
    ax.set_xlabel("Universe size N (qubits)")
    ax.set_ylabel("Median gap to brute-force optimum (%)")
    ax.set_title("Experiment C-HPC: QAOA solution quality vs N")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=Path, default=Path("results/hpc_qaoa.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    df = load_jsonl(args.in_file)
    if df.empty:
        print(f"No rows in {args.in_file}")
        return 1
    summary = summarise(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "experiment_c_hpc_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    plot_scaling(summary, figures_dir / "exp_c_hpc_scaling.png")
    plot_quality(summary, figures_dir / "exp_c_hpc_quality.png")

    pd.options.display.float_format = "{:.4f}".format
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
