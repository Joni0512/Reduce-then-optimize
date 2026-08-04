"""GNN architecture screening chart - mean best-val service rate (+/- std
across seeds) per method/layer combination, SIL bi200/ss100 class1 legacy.
Grouped bar chart, MLP baseline highlighted separately from GNN variants."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#2a78d6"
AMBER = "#e69f00"  # Okabe-Ito orange, colorblind-safe against BLUE
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# seed -> best val service rate, per method/layer combination
results = {
    "MLP": {1: 0.7390, 2: 0.7107, 3: 0.7516, 4: 0.7610, 5: 0.7013},
    "GCN\nL1": {2: 0.7390, 3: 0.7327, 4: 0.7484, 5: 0.7233, 6: 0.6887},
    "Mean\nL1": {2: 0.7296, 3: 0.7170, 4: 0.7264, 5: 0.7264, 6: 0.7327},
    "Mean\nL2": {2: 0.6981, 3: 0.7233, 4: 0.7327, 5: 0.7421, 6: 0.7358},
    "GCN\nL2": {2: 0.7358, 3: 0.7201, 4: 0.7296, 5: 0.7138, 6: 0.7233},
    "Pool\nL2": {1: 0.7296, 2: 0.7075, 3: 0.7233, 4: 0.7107, 5: 0.7327},
    "Pool\nL1": {1: 0.6887, 2: 0.7327, 3: 0.7138, 4: 0.7044, 5: 0.7296, 6: 0.7138},
}

labels = list(results.keys())
means = [np.mean(list(v.values())) for v in results.values()]
stds = [np.std(list(v.values()), ddof=1) for v in results.values()]
ns = [len(v) for v in results.values()]
colors = [AMBER if lbl == "MLP" else BLUE for lbl in labels]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(labels))
bar_width = 0.6
bars = ax.bar(
    x, means, bar_width, yerr=stds, capsize=4,
    color=colors, edgecolor="#fcfcfb", linewidth=1.5,
    error_kw={"ecolor": TEXT_SECONDARY, "linewidth": 1.3},
    zorder=3,
)

# selective direct labels: mean value above each bar (above the error whisker)
for xi, mean, std, n in zip(x, means, stds, ns):
    ax.annotate(
        f"{mean*100:.1f}%",
        (xi, mean + std), textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=10, color=TEXT_PRIMARY,
    )
    ax.annotate(
        f"n={n}",
        (xi, 0.01), textcoords="offset points", xytext=(0, 0),
        ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY,
    )

ax.set_ylabel("Best Val Service Rate", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "GNN Architecture Screening — Best Val Service Rate by Method\n"
    "SIL bi200/ss100 class1 legacy, mean $\\pm$ std across seeds",
    fontsize=12.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color=TEXT_SECONDARY)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

# legend: MLP baseline vs. GNN variants (color = identity of baseline vs. not)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=AMBER, label="MLP (baseline)"),
    Patch(facecolor=BLUE, label="GNN variant"),
]
ax.legend(
    handles=legend_handles, loc="upper right", frameon=False,
    fontsize=9.5, labelcolor=TEXT_SECONDARY,
)

plt.tight_layout()
out_path = "figures_export/gnn_architecture_screening_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gnn_architecture_screening_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
