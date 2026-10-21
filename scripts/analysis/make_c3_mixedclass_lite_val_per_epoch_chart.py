"""
2026-10-19: c3 MIXEDCLASS-LITE, batch_interval=400/step_size=80 - pooled val service rate per epoch (seed 42, epochs 5,
learning_rate 0.0001, legacy scoring, lite split = 34 train files without lc203/lr203/lr204/lr206/lr207, 12 val files = 623
requests, optimal_val pre-pass skipped). Counts (n/623) from the "Epoch k: pooled val service rate" lines of the cluster logs
sil_training_bi400_ss80_c3_mixedclass_lite_*_seed42.log (pasted into the chat on 2026-10-19):
  baseline (still running)  475 (epoch 1 only)                pair-only  472 482 482 485 489 (best 78.49 %, epoch 5)
  Request-only t=0.6 195 206 221 258 285 | t=0.7 200 217 227 249 275 | t=0.8 200 201 210 215 217 | t=0.9 196 (epoch 1, running)
  Combined (Pair 0.5) t=0.6 203 215 235 246 260 | t=0.7 203 203 235 252 266 | t=0.8 203 213 199 213 165 | t=0.9 197 191 176 178 172
Left panel Request-only, right panel Combined; baseline and pair-only are drawn in both as references.
Run: ./venv/bin/python3 scripts/analysis/make_c3_mixedclass_lite_val_per_epoch_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
N = 623
NAVY, ORANGE = "#1E2761", "#E8A33D"
SHADES = {"0.6": "#7FBF9F", "0.7": "#3E8F6C", "0.8": "#2A6B4D", "0.9": "#14402D"}
REF = {"Baseline (running, epoch 1 of 5)": ([475], NAVY, "--"), "Pair Pruner only": ([472, 482, 482, 485, 489], ORANGE, "-")}
REQ = {"0.6": [195, 206, 221, 258, 285], "0.7": [200, 217, 227, 249, 275], "0.8": [200, 201, 210, 215, 217], "0.9": [196]}
COMB = {"0.6": [203, 215, 235, 246, 260], "0.7": [203, 203, 235, 252, 266], "0.8": [203, 213, 199, 213, 165], "0.9": [197, 191, 176, 178, 172]}
OFFS = {"Request only": {"0.6": 8, "0.7": -3, "0.8": 5, "0.9": -3}, "Combined": {"0.6": -9, "0.7": 8, "0.8": -9, "0.9": 5}}


def panel(ax, title, data, name):
    for lab, (counts, color, ls) in REF.items():
        xs = range(1, len(counts) + 1)
        ys = [100 * c / N for c in counts]
        ax.plot(xs, ys, marker="o" if len(counts) == 1 else "s", color=color, linewidth=2.2, markersize=8 if len(counts) == 1 else 5,
                linestyle=ls, label=lab, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0), fontsize=8.5, color=color,
                    va="center", fontweight="bold")
    for t, counts in data.items():
        xs = list(range(1, len(counts) + 1))
        ys = [100 * c / N for c in counts]
        running = len(counts) < 5
        ax.plot(xs, ys, marker="o", color=SHADES[t], linewidth=2, markersize=6, label=f"{name} t={t}" + (" (running)" if running else ""),
                linestyle="--" if running else "-", zorder=3)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, OFFS[name][t]), fontsize=8.5,
                    color=SHADES[t], va="center", fontweight="bold")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.7, 5.8)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, linestyle=":", alpha=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.3, frameon=False, ncol=2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    panel(axes[0], "Request Pruner only (Request threshold t)", REQ, "Request only")
    panel(axes[1], "Request + Pair Pruner (Pair t=0.5, Request threshold t)", COMB, "Combined")
    axes[0].set_ylabel("Pooled val service rate (%)", fontsize=10.5)
    fig.suptitle("c3 mixedclass-lite, batch_interval=400/step_size=80: val service rate per epoch (seed 42, epochs 5, lr 0.0001) - status 2026-10-19",
                 fontsize=12.2)
    fig.text(0.5, 0.005, "Lite split: 34 train files (no lc203, lr203, lr204, lr206, lr207), 12 val files = 623 requests (1 request = 0.16 pp). "
             "Dashed = still running. Not comparable to the full-split runs.", ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_lite_bi400ss80_val_per_epoch.{ext}", dpi=200)
    print("saved c3_mixedclass_lite_bi400ss80_val_per_epoch.png/.pdf")


if __name__ == "__main__":
    main()
