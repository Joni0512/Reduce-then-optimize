"""GCN L1 HP screening heatmap - best val service rate across learning_rate x
hidden_dim, dropout=0.0 fixed, seed 1 (single-seed coarse screening).
SIL bi200/ss100 class1 legacy."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

learning_rates = ["0.0001\n(Table 3)", "0.0003", "0.001"]
hidden_dims = ["64", "128", "256"]
# rows = learning rate (top->bottom: 0.0001, 0.0003, 0.001), cols = hidden_dim
values = np.array([
    [0.7327, 0.7327, 0.7296],
    [0.7516, 0.7547, 0.7673],
    [0.7673, 0.7799, 0.7767],
])

# sequential single hue, light -> dark (BLUE ramp), per dataviz convention
cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#eaf1fb", "#2a78d6", "#0d2f56"]
)

fig, ax = plt.subplots(figsize=(6.4, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

im = ax.imshow(values, cmap=cmap, vmin=0.72, vmax=0.79, aspect="auto")

ax.set_xticks(range(len(hidden_dims)))
ax.set_xticklabels(hidden_dims, fontsize=11, color=TEXT_SECONDARY)
ax.set_yticks(range(len(learning_rates)))
ax.set_yticklabels(learning_rates, fontsize=11, color=TEXT_SECONDARY)
ax.set_xlabel("Hidden Dim", fontsize=11.5, color=TEXT_SECONDARY)
ax.set_ylabel("Learning Rate", fontsize=11.5, color=TEXT_SECONDARY)
ax.set_title(
    "GCN L1 HP Screening — Best Val Service Rate\n"
    "SIL bi200/ss100 class1 legacy, dropout=0.0, seed=1 (single-seed screening)",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

# direct labels on every cell (small grid, all values are the point)
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        v = values[i, j]
        # readable label color depending on cell darkness
        label_color = "#fcfcfb" if v > 0.755 else TEXT_PRIMARY
        weight = "bold" if v == values.max() else "normal"
        ax.text(
            j, i, f"{v*100:.2f}%", ha="center", va="center",
            fontsize=13, color=label_color, fontweight=weight,
        )

ax.set_xticks(np.arange(-0.5, len(hidden_dims), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(learning_rates), 1), minor=True)
ax.grid(which="minor", color="#fcfcfb", linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
cbar.outline.set_visible(False)
cbar.set_label("Best Val Service Rate", fontsize=10, color=TEXT_SECONDARY)
cbar.ax.yaxis.set_major_formatter(
    plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
)

plt.tight_layout()
out_path = "figures_export/gcn_l1_hp_screening_heatmap.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gcn_l1_hp_screening_heatmap.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
