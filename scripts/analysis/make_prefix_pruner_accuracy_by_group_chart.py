"""
2026-10-19: chart of outputs/prefix_pruner_accuracy_by_group.csv (written by analyze_prefix_pruner_accuracy_by_group.py):
own accuracy of the final c2 expert-prefix-label Request Pruner per class x geometry group, threshold 0.5, min_retention 0.3,
12 mixedclass test instances. Left: share of the reference-route prefix targets (label 1) that the pruner removes, all windows
(replay, approximate from window 1 on). Right: share of vehicles that lose at least one of their first K=2 requests in window 0
(exact state). Class-2 groups rest on only 5-6 vehicles in window 0 (right panel).
Run: ./venv/bin/python3 scripts/analysis/make_prefix_pruner_accuracy_by_group_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
CFG_LABEL = {"bi200_ss100": "batch_interval=200/step_size=100", "bi400_ss200": "batch_interval=400/step_size=200",
             "bi400_ss80": "batch_interval=400/step_size=80"}
COLOR = {"bi200_ss100": "#276575", "bi400_ss200": "#A6493A", "bi400_ss80": "#3E8F6C"}
GROUPS = [("class1", "lc"), ("class1", "lr"), ("class1", "lrc"), ("class2", "lc"), ("class2", "lr"), ("class2", "lrc")]


def panel(ax, df, col, ylabel, title):
    x = np.arange(len(GROUPS))
    w = 0.26
    for i, cfg in enumerate(CFG_LABEL):
        vals = []
        for c, g in GROUPS:
            r = df[(df.config == cfg) & (df["class"] == c) & (df.geometry == g)]
            vals.append(float(r[col].iloc[0]))
        ax.bar(x + (i - 1) * w, vals, width=w, color=COLOR[cfg], zorder=3, label=CFG_LABEL[cfg])
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c[-1]} {g}" for c, g in GROUPS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def main():
    df = pd.read_csv("outputs/prefix_pruner_accuracy_by_group.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    panel(axes[0], df, "label1_pruned_%_all_windows", "Prefix targets removed by the pruner (%)",
          "Share of reference-route prefix targets (label 1) pruned, all windows")
    panel(axes[1], df, "lost_any_of_K_%", "Vehicles losing a target (%)",
          "Vehicles that lose at least one of their first 2 requests, window 0")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("c2 expert-prefix-label Request Pruner (t=0.5, min_retention 0.3): own accuracy per instance group", fontsize=12.5)
    fig.text(0.5, 0.005, "12 mixedclass test instances (2 per group), never seen by the pruner. Right panel: 19-21 vehicles per class-1 group, "
             "only 5-6 per class-2 group (one vehicle = 17-20 pp). Left panel: replay of the RHO-baseline windows (approximate from window 1).",
             ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c2_prefix_pruner_accuracy_by_group.{ext}", dpi=200)
    print("saved c2_prefix_pruner_accuracy_by_group.png/.pdf")


if __name__ == "__main__":
    main()
