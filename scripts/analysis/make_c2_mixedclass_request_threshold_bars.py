"""
2026-10-10: c2 MIXEDCLASS, Request Pruner only (expert-prefix-label MLP, min_retention 0.3):
grouped BAR chart comparing the thresholds t = 0.3 ... 0.8 against the no-pruner baseline.
Service rate = mean of seeds 42 and 1 (same two seeds for every bar), dots = the two seeds,
pruning rate (share of requests removed, from compute_pruning_rates.py) printed under the value.

Per-seed "Best val service rate" (percent), seeds 42 / 1 (sil_training_{cfg}_mixedclass_requestpruner_
prefixlabel_t{t}_seed{s}.log; baseline: ..._mixedclass_requestpruner_baseline_seed{s}.log):
  bi200/ss100: base 60.67/67.58  t0.3 45.26/56.50  t0.4 37.08/42.54  t0.5 35.31/33.71  t0.7 33.39/34.19  t0.8 30.98/34.51
  bi400/ss200: base 43.50/47.51  t0.3 50.72/47.35  t0.5 49.44/49.28  t0.7 48.48/44.94  t0.8 45.10/43.66
  bi400/ss80 : base 49.44/57.14  t0.3 32.42/35.15  t0.5 36.44/37.40  t0.7 33.39/31.62  t0.8 30.98/30.82
Pruning rate (%): bi200/ss100 29.9/34.8/39.1/45.3/47.9; bi400/ss200 32.1/-/35.5/38.3/39.1;
bi400/ss80 35.8/-/42.8/48.6/50.9 (t0.3/0.4/0.5/0.7/0.8). t=0.1/0.2/0.4 added 2026-10-12, t=0.6 added 2026-10-13 (service seeds 42/1: bi200/ss100 34.35/35.47, bi400/ss200 50.56/47.19, bi400/ss80 33.71/31.46; pruning 42.1/36.9/46.0 %).
HPs: epochs 5, lr 0.0001, legacy scoring, pickup_slack off.

Run: ./venv/bin/python3 scripts/analysis/make_c2_mixedclass_request_threshold_bars.py
"""
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
BARS = ["Baseline", "t=0.1", "t=0.2", "t=0.3", "t=0.4", "t=0.5", "t=0.6", "t=0.7", "t=0.8"]
COLORS = ["#1E2761", "#D9EEE5", "#BFE3D3", "#A9D6C4", "#7FBF9F", "#3E8F6C", "#34795A", "#2A6B4D", "#14402D"]

# 2026-10-10: t=0.1/0.2/0.4 added for all configs (seeds 42/1, from the cluster logs of 2026-10-12)
# 2026-10-13: t=0.6 added (seeds 42/1; Best val from the logs of 2026-10-12, pruning rate = mean of the two seeds'
# compute_pruning_rates.py output: bi200/ss100 42.09/42.12, bi400/ss200 36.92/36.93, bi400/ss80 45.96/46.03 %)
# (seed values [42, 1], pruning rate %)
DATA = {
    CONFIGS[0]: [([60.67, 67.58], 0), ([56.18, 57.46], 11.4), ([51.36, 57.95], 24.4), ([45.26, 56.50], 29.9),
                 ([37.08, 42.54], 34.8), ([35.31, 33.71], 39.1), ([34.35, 35.47], 42.1), ([33.39, 34.19], 45.3), ([30.98, 34.51], 47.9)],
    CONFIGS[1]: [([43.50, 47.51], 0), ([48.80, 47.35], 22.7), ([50.72, 47.67], 28.9), ([50.72, 47.35], 32.1),
                 ([50.72, 47.67], 34.0), ([49.44, 49.28], 35.5), ([50.56, 47.19], 36.9), ([48.48, 44.94], 38.3), ([45.10, 43.66], 39.1)],
    CONFIGS[2]: [([49.44, 57.14], 0), ([47.35, 46.87], 23.6), ([50.24, 50.08], 31.7), ([32.42, 35.15], 35.8),
                 ([40.29, 41.25], 39.3), ([36.44, 37.40], 42.8), ([33.71, 31.46], 46.0), ([33.39, 31.62], 48.6), ([30.98, 30.82], 50.9)],
}


def main():
    fig, ax = plt.subplots(figsize=(16, 6.8))
    x = np.arange(len(CONFIGS))
    n, w = len(BARS), 0.1
    for i, (name, color) in enumerate(zip(BARS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            d = DATA[cfg][i]
            if d is None:
                ax.annotate("n/a", (xi + offs, 0.6), ha="center", fontsize=8, color="#888888")
                continue
            seeds, prune = d
            m = st.mean(seeds)
            ax.bar(xi + offs, m, width=w, color=color, zorder=3, label=name)
            ax.scatter([xi + offs] * len(seeds), seeds, s=16, color="white", edgecolor="#222222",
                       linewidth=0.7, zorder=5)
            ax.annotate(f"{m:.1f}", (xi + offs, max(seeds)), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, fontweight="bold")
            # 2026-10-10: pruning rate inside the bar (vertical) - avoids overlap with the neighbouring value labels
            if i > 0:
                ax.text(xi + offs, 1.5, f"{prune:.0f}% pruned", rotation=90, ha="center", va="bottom",
                        fontsize=8, color=("#222222" if i <= 3 else "white"), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10.5)
    ax.set_ylabel("Best val service rate (%)", fontsize=11)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("c2 mixedclass, Request Pruner only: service rate per threshold (t = 0.1 ... 0.8, incl. 0.6) vs. baseline", fontsize=12)
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False, ncol=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "c2, mixedclass, seeds 42 and 1 (dots = single seeds, bar = mean), epochs=5, lr=0.0001, "
             "pickup_slack off, Request Pruner min_retention 0.3", ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / "c2_mixedclass_request_threshold_bars.png", dpi=200)
    fig.savefig(OUT_DIR / "c2_mixedclass_request_threshold_bars.pdf")
    print("saved c2_mixedclass_request_threshold_bars.png/.pdf")


if __name__ == "__main__":
    main()
