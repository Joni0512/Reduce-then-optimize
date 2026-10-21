"""
2026-10-14: ONE joint chart (service rate) and ONE joint chart (runtime) for the three configs of the class-1 Request/Pair
Pruner comparison (c2): Baseline vs. Request Pruner only vs. Request + Pair Pruner, bars side by side per config, mean +- std
over seeds. Merges the values of the three per-config charts so they stay identical to those:
  make_bi200ss100_combined_pruner_chart.py, make_bi400ss200_combined_pruner_chart.py, make_bi400ss80_combined_pruner_chart.py
Values (all "Best val service rate" / log runtimes, threshold 0.5, min_retention_fraction 0.3, epochs 5, lr 0.0001, legacy scoring):
  service rate [mean, std]: bi200/ss100  baseline 74.5/1.8   request 65.85/2.45  combined 64.40/3.57
                            bi400/ss200  baseline 51.8/1.6   request 55.91/1.66  combined 56.23/1.16
                            bi400/ss80   baseline 79.32/0.64 request 82.58/0.72  combined 82.20/0.96
  runtime minutes [mean, std]: bi200/ss100 95.6/-  49.4/1.4  37.0/1.7 ; bi400/ss200 79.3/-  67.9/1.6  36.5/1.6 ;
                               bi400/ss80 107.5/4.0  63.9/1.1  44.5/1.7
Seeds: n=5 (42,1,2,3,4) everywhere except the baseline of bi400/ss80 (n=4, seeds 42,1,2,3). Combined runtimes use the cluster
seeds only (seed 42 ran locally, not comparable); baseline runtimes of bi200/ss100 and bi400/ss200 are earlier cited values
without std (see the per-config scripts for the provenance).
Run: ./venv/bin/python3 scripts/analysis/make_c2_class1_combined_pruner_all_configs_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
VARIANTS = ["No pruner (baseline)", "Request Pruner only", "Request + Pair Pruner"]
COLORS = ["#1E2761", "#276575", "#3E8F6C"]

# (mean, std) per config and variant; std None = not available
SERVICE = {
    CONFIGS[0]: [(74.5, 1.8), (65.85, 2.45), (64.40, 3.57)],
    CONFIGS[1]: [(51.8, 1.6), (55.91, 1.66), (56.23, 1.16)],
    CONFIGS[2]: [(79.32, 0.64), (82.58, 0.72), (82.20, 0.96)],
}
RUNTIME = {
    CONFIGS[0]: [(95.6, None), (49.4, 1.4), (37.0, 1.7)],
    CONFIGS[1]: [(79.3, None), (67.9, 1.6), (36.5, 1.6)],
    CONFIGS[2]: [(107.5, 4.0), (63.9, 1.1), (44.5, 1.7)],
}


def chart(data, ylabel, title, stem, ylim, fmt, pct=False):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(CONFIGS))
    w = 0.26
    for i, (var, color) in enumerate(zip(VARIANTS, COLORS)):
        offs = (i - 1) * w
        for xi, cfg in zip(x, CONFIGS):
            m, s = data[cfg][i]
            ax.bar(xi + offs, m, width=w, color=color, zorder=3, label=var,
                   yerr=(s if s is not None else 0), capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#222222"))
            lab = fmt.format(m) + (f" ± {fmt.format(s)}" if s is not None else "")
            ax.annotate(lab, (xi + offs, m + (s or 0)), textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title(title, fontsize=12.5)
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.text(0.5, 0.005, "c2, class 1, mean ± std over n=5 seeds (baseline batch_interval=400/step_size=80: n=4); Request Pruner "
             "t=0.5, Pair Pruner GNNv1 t=0.5, min_retention_fraction 0.3; epochs=5, lr=0.0001, legacy scoring",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"saved {OUT_DIR / stem}.png/.pdf")


if __name__ == "__main__":
    chart(SERVICE, "Best val service rate (%)",
          "c2 class 1: Baseline vs. Request Pruner vs. Request + Pair Pruner - Service Rate (all configs)",
          "c2_class1_combined_pruner_all_configs_service_rate", (0, 100), "{:.1f}", pct=True)
    chart(RUNTIME, "Runtime (minutes, full 5-epoch training run)",
          "c2 class 1: Baseline vs. Request Pruner vs. Request + Pair Pruner - Runtime (all configs)",
          "c2_class1_combined_pruner_all_configs_runtime", (0, 130), "{:.1f}")
