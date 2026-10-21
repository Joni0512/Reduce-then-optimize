"""
2026-10-18: INTERIM chart - pooled val service rate per epoch of the c3 mixedclass runs (seed 42, epochs 5, lr 0.0001, legacy
scoring, pickup_slack on, 12 val files = 623 requests). Values are counts out of 623 from the "Epoch k: pooled val service
rate = x (n/623 across 12 files)" log lines (cluster logs pasted into the chat on 2026-10-18).
Left panel: batch_interval=200/step_size=100, FULL split (39 train files) - 3-epoch baseline (job 5657327) vs. the 5-epoch variants.
  baseline  326 414 449        pair-only 296 376 419 411 424        Combined t=0.5  197 209 206 252 282
  Combined t=0.7  21 192 191 195 195      t=0.8  0 183 193 183 183      t=0.9  76 140 146 154 154
Right panel: batch_interval=400/step_size=80, LITE split (34 train files, no lc203/lr203/lr204/lr206/lr207, optimal_val pre-pass
skipped) - runs still in progress on 2026-10-18 ~13:00: Pair-only 472 482 482 (epoch 4 running), Combined t=0.6 203 215 235 246 260
(done), t=0.7 203 203 235 252 266 (done), t=0.8 203 213 (epoch 3 running); the lite baseline has no epoch value yet (epoch 1 nearly done).
Run: ./venv/bin/python3 scripts/analysis/make_c3_mixedclass_val_per_epoch_interim_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N = 623

# name -> (counts per epoch, finished?, color)
LEFT = {
    "Baseline (3 epochs only)": ([326, 414, 449], True, "#1E2761"),
    "Pair Pruner only": ([296, 376, 419, 411, 424], True, "#E8A33D"),
    "Combined, Request t=0.5": ([197, 209, 206, 252, 282], True, "#7FBF9F"),
    "Combined, Request t=0.7": ([21, 192, 191, 195, 195], True, "#3E8F6C"),
    "Combined, Request t=0.8": ([0, 183, 193, 183, 183], True, "#2A6B4D"),
    "Combined, Request t=0.9": ([76, 140, 146, 154, 154], True, "#14402D"),
}
RIGHT = {
    "Pair Pruner only (running, epoch 4)": ([472, 482, 482], False, "#E8A33D"),
    "Combined, Request t=0.6 (done)": ([203, 215, 235, 246, 260], True, "#5BAF8A"),
    "Combined, Request t=0.7 (done)": ([203, 203, 235, 252, 266], True, "#3E8F6C"),
    "Combined, Request t=0.8 (running, epoch 3)": ([203, 213], False, "#2A6B4D"),
}


OFFSETS = {"Combined, Request t=0.7": 7, "Combined, Request t=0.8": -7,
           "Combined, Request t=0.6 (done)": -8, "Combined, Request t=0.7 (done)": 8}


def panel(ax, data, title, note=None, legend_loc="lower right", legend_ncol=1):
    for name, (counts, done, color) in data.items():
        xs = list(range(1, len(counts) + 1))
        ys = [100 * c / N for c in counts]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=6, label=name, zorder=3,
                linestyle="-" if done else "--")
        # 2026-10-18: end labels of nearby curves are offset vertically (they overlapped at epoch 5)
        dy = OFFSETS.get(name, 0)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, dy), fontsize=8.5,
                    color=color, va="center", fontweight="bold")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Epoch", fontsize=10.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.7, 5.8)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, linestyle=":", alpha=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc=legend_loc, fontsize=8.3, frameon=False, ncol=legend_ncol)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8.5, va="top", color="#A6493A")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    panel(axes[0], LEFT, "batch_interval=200/step_size=100, full split (39 train files)", legend_loc="upper right", legend_ncol=2)
    panel(axes[1], RIGHT, "batch_interval=400/step_size=80, lite split (34 train files) - INTERIM",
          note="Lite baseline: epoch 1 not finished yet")
    axes[0].set_ylabel("Pooled val service rate (%)", fontsize=10.5)
    fig.suptitle("c3 mixedclass: val service rate per epoch (seed 42, epochs 5, lr 0.0001) - interim status 2026-10-18", fontsize=12.5)
    fig.text(0.5, 0.005, "12 val files = 623 requests (1 request = 0.16 pp). Dashed = run still in progress. The lite split is not "
             "comparable to the full split. Left baseline was run for 3 epochs only (7-day limit).",
             ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_val_per_epoch_interim.{ext}", dpi=200)
    print("saved", OUT_DIR / "c3_mixedclass_val_per_epoch_interim.png/.pdf")


if __name__ == "__main__":
    main()
