"""Pool L2 Bayesian sweep chart (preliminary) - best_val_service_rate vs.
learning_rate, marker color = hidden_dim. SIL bi200/ss100 class1 legacy,
dropout=0.0, seed=1. hidden_dim=256 trials excluded (still re-running after
SLURM time-limit cancellations, see make_gcn_l1_bayes_sweep_chart.py for the
sibling GCN L1 chart this mirrors)."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"
HIDDEN_DIM_COLORS = {64: "#e69f00", 128: "#2a78d6"}

runs = [
    ("crimson-sweep-1", 0.001869, 128, 0.7642, True),
    ("fiery-sweep-7", 0.000943, 128, 0.7610, False),
    ("olive-sweep-4", 0.001105, 128, 0.7547, False),
    ("feasible-sweep-9", 0.000555, 128, 0.7547, False),
    ("vocal-sweep-6", 0.004709, 64, 0.7516, False),
    ("exalted-sweep-5", 0.000420, 64, 0.7358, False),
]

fig, ax = plt.subplots(figsize=(7.2, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for hd, color in HIDDEN_DIM_COLORS.items():
    xs = [lr for _, lr, h, sr, best in runs if h == hd]
    ys = [sr for _, lr, h, sr, best in runs if h == hd]
    ax.scatter(
        xs, ys, s=90, color=color, edgecolor="#fcfcfb", linewidth=1.2,
        zorder=3, label=f"hidden_dim={hd}",
    )

best_lr, best_hd, best_sr = 0.001869, 128, 0.7642
ax.scatter(
    [best_lr], [best_sr], s=260, facecolors="none",
    edgecolors=TEXT_PRIMARY, linewidth=1.6, zorder=4,
)
ax.annotate(
    f"best so far: {best_sr*100:.2f}%\nlr={best_lr:.4f}, h={best_hd}",
    (best_lr, best_sr), textcoords="offset points", xytext=(10, -22),
    ha="left", fontsize=9.5, color=TEXT_PRIMARY,
)

ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Best Val Service Rate", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "Pool L2 Bayesian HP Sweep — Service Rate vs. Learning Rate (preliminary)\n"
    "SIL bi200/ss100 class1 legacy, dropout=0.0, seed=1 — hidden_dim=256 pending",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0.70, 0.80)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", color=GRID, linewidth=1, zorder=0, which="both")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)

plt.tight_layout()
out_path = "figures_export/pool_l2_bayes_sweep_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/pool_l2_bayes_sweep_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
