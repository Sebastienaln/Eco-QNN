#!/usr/bin/env python3
"""
Plotting script for re-uploading study results.
Reads Excel file with raw results and plots mean ± std accuracy vs layers.

Usage:
  python plot_reupload_results.py --input RESULTS/reupload_study_batch.xlsx --output plot.png
  python plot_reupload_results.py --help
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: Path):
    """Load raw results from Excel and compute summary statistics."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    df_raw = pd.read_excel(path, sheet_name="raw")
    
    # Compute summary statistics
    summary = df_raw.groupby(["dimension", "layers"]).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        n_runs=("accuracy", "size"),
    ).reset_index()
    
    return summary


def plot_summary(summary: pd.DataFrame, output: Path, show: bool = False):
    """Plot mean ± std accuracy vs layers for each dimension."""
    dims = sorted(summary["dimension"].unique())
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    for d in dims:
        df_d = summary[summary["dimension"] == d]
        ax.errorbar(
            df_d["layers"],
            df_d["mean_accuracy"],
            yerr=df_d["std_accuracy"],
            fmt="o-",
            capsize=5,
            capthick=1.5,
            markersize=6,
            linewidth=1.5,
            label=f"Dimension {d} (n={df_d['n_runs'].iloc[0]:.0f})",
        )
    
    ax.set_xlabel("Number of Re-uploading Layers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Re-uploading Circuit Study: Accuracy vs Layers (mean ± std)", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim([0, 1.05])
    
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Plot saved to {output}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot re-uploading study results with error bars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input", type=Path, default=Path("RESULTS/reupload_study_batch.xlsx"),
                   help="Path to Excel file from run_reupload_study.py")
    p.add_argument("--output", type=Path, default=Path("RESULTS/reupload_study_plot.png"),
                   help="Output PNG file path")
    p.add_argument("--show", action="store_true",
                   help="Display plot in window (useful for local runs)")
    return p.parse_args()


def main():
    args = parse_args()
    summary = load_results(args.input)
    
    print("Summary statistics:")
    print(summary.to_string())
    print()
    
    plot_summary(summary, args.output, show=args.show)


if __name__ == "__main__":
    main()
