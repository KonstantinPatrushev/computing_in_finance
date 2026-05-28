"""Build a combined synthetic + bootstrap-realistic Experiment A summary.

Loads two JSONL files (synthetic and bootstrap-realistic), aggregates each
into a per-(N, K, solver) median table, and writes:

* ``results/final/experiment_a_combined.csv`` — long-format table with a
  ``source`` column distinguishing the two data generators.
* ``results/final/figures/exp_a_scalability_combined.png`` — two-panel plot
  comparing wall time scaling on synthetic vs bootstrap-realistic Σ.
* ``results/final/figures/exp_a_quality_combined.png`` — analogous panels
  for gap-to-best.

The point of the two-source comparison is to show that Tabu's speedup
holds on realistic financial covariance (block structure, low condition
number) — not just on dense random Σ where SCIP happens to struggle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Re-use the loader from the existing aggregator (small file, easier to import).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_experiment_a import add_approx_ratio, load_jsonl, summarise


COLORS = {"scip": "tab:blue", "ecos_bb": "tab:orange", "neal": "tab:green", "tabu": "tab:red"}
MARKERS = {"scip": "o", "ecos_bb": "s", "neal": "^", "tabu": "D"}
SOURCE_LABEL = {
    "synthetic": "Dense random Σ (A·Aᵀ)",
    "bootstrap-sp500": "Bootstrap-realistic Σ (S&P 500 Ledoit–Wolf)",
}


def build_combined(synth_path: Path, boot_path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in (synth_path, boot_path):
        if not path.exists():
            print(f"missing {path}; skipping")
            continue
        df = load_jsonl(path)
        df = add_approx_ratio(df)
        parts.append(df)
    if not parts:
        raise SystemExit("no input files found")
    return pd.concat(parts, ignore_index=True)


def plot_two_panel(summary: pd.DataFrame, value_col: str, ylabel: str, log_y: bool, out_path: Path) -> None:
    sources = [s for s in SOURCE_LABEL if s in summary["source"].unique()]
    n_panels = len(sources)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, source in zip(axes, sources):
        sub_src = summary[summary["source"] == source]
        for solver, sub in sub_src.groupby("solver"):
            sub_sorted = sub.sort_values("N")
            ax.plot(
                sub_sorted["N"],
                sub_sorted[value_col],
                marker=MARKERS.get(solver, "x"),
                label=solver,
                color=COLORS.get(solver),
                linewidth=2,
                markersize=7,
            )
        ax.set_title(SOURCE_LABEL.get(source, source))
        ax.set_xlabel("Universe size N")
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=Path("results/experiment_a_with_tabu.jsonl"))
    parser.add_argument("--bootstrap", type=Path, default=Path("results/experiment_a_bootstrap.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = build_combined(args.synthetic, args.bootstrap)
    summary = summarise(df)
    out_csv = args.out_dir / "experiment_a_combined.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    plot_two_panel(
        summary,
        value_col="wall_time_median",
        ylabel="Wall time (s, log)",
        log_y=True,
        out_path=figures_dir / "exp_a_scalability_combined.png",
    )
    plot_two_panel(
        summary,
        value_col="gap_median_pct",
        ylabel="Gap to best (%)",
        log_y=False,
        out_path=figures_dir / "exp_a_quality_combined.png",
    )

    pd.options.display.float_format = "{:.3f}".format
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
