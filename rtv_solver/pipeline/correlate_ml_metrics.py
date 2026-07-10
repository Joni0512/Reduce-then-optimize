"""
Correlates the pruner's classifier metrics (accuracy, precision, recall, F1-F3,
edge_reduction - from evaluate_pruner_metrics_all.py) against the actual solver
outcome (service_rate_pct_change, total_time_pct_change - from
parse_card2_firstcheck.py), per (gnn_version, instance, threshold).

Produces one correlation-matrix heatmap per pipeline (COAML, RH) plus a combined
one, so it's visible at a glance whether "the pruner scored well as a classifier"
actually predicts "the solver ran faster / serviced more requests" - or not.

Usage:
    python -m rtv_solver.pipeline.correlate_ml_metrics
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "outputs" / "outputs" / "new_tests" / "card2_firstcheck" / "analysis"
PRUNER_METRICS_CSV = ANALYSIS_DIR / "pruner_metrics" / "pruner_metrics_long.csv"
SOLVER_METRICS_CSV = ANALYSIS_DIR / "card2_firstcheck_long.csv"
OUT_DIR = ANALYSIS_DIR / "ml_metric_correlation"

ML_METRICS = ["accuracy", "precision", "recall", "f1", "f2", "f3", "edge_reduction"]
SOLVER_METRICS = ["service_rate_pct_change", "total_time_pct_change"]


def load_merged() -> pd.DataFrame:
    pruner = pd.read_csv(PRUNER_METRICS_CSV)
    solver = pd.read_csv(SOLVER_METRICS_CSV)

    # Only gnn_v1/gnn_v2 rows have a pruner threshold to join on; baseline has none
    # and cardinality must be 2 to match the pruner_metrics sweep (mc3 not covered there).
    solver = solver[solver["method"].isin(["gnn_v1", "gnn_v2"]) & (solver["cardinality"] == 2)]

    merged = solver.merge(
        pruner,
        left_on=["method", "instance", "threshold"],
        right_on=["gnn_version", "instance", "threshold"],
        how="inner",
        suffixes=("", "_pruner"),
    )
    return merged


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ML_METRICS + SOLVER_METRICS
    return df[cols].corr()


def plot_heatmap(corr: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    # Separator line between ML metrics and solver-outcome metrics, in both directions.
    split = len(ML_METRICS) - 0.5
    ax.axhline(split, color="black", linewidth=1.2)
    ax.axvline(split, color="black", linewidth=1.2)

    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / filename}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    merged.to_csv(OUT_DIR / "ml_metrics_solver_merged.csv", index=False)

    combined_corr = corr_matrix(merged)
    combined_corr.to_csv(OUT_DIR / "corr_matrix_combined.csv")
    plot_heatmap(
        combined_corr,
        "ML metrics vs. solver outcome - both pipelines pooled",
        "corr_matrix_combined.png",
    )

    for pipeline in ["coaml", "rh"]:
        sub = merged[merged["pipeline"] == pipeline]
        corr = corr_matrix(sub)
        corr.to_csv(OUT_DIR / f"corr_matrix_{pipeline}.csv")
        plot_heatmap(
            corr,
            f"ML metrics vs. solver outcome - {pipeline.upper()} (n={len(sub)})",
            f"corr_matrix_{pipeline}.png",
        )

    print("\nCorrelation of each ML metric with solver outcome (combined, both pipelines):")
    print(combined_corr.loc[ML_METRICS, SOLVER_METRICS].round(3).to_string())

    print("\nSaved merged data + matrices to:", OUT_DIR)


if __name__ == "__main__":
    main()
