import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

epochs = [1, 2, 3, 4, 5]
lr001 = [18.36, 5.77, 51.00, 56.98, 42.18]
lr0001 = [52.47, 64.95, 69.05, 67.89, 67.05]

fig, ax = plt.subplots(figsize=(8.6, 4.9))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    epochs, lr001, color=ORANGE, linewidth=2, marker="s", markersize=7,
    markerfacecolor=ORANGE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="LR = 0.001",
)
ax.plot(
    epochs, lr0001, color=BLUE, linewidth=2, marker="o", markersize=7,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="LR = 0.0001 (Table 3)",
)

best_lr001 = epochs[lr001.index(max(lr001))]
best_lr0001 = epochs[lr0001.index(max(lr0001))]
ax.annotate(f"best {max(lr001):.2f}%", (best_lr001, max(lr001)),
            textcoords="offset points", xytext=(15, -18), ha="left", fontsize=9.5,
            color=ORANGE, fontweight="bold")
ax.annotate(f"best {max(lr0001):.2f}%", (best_lr0001, max(lr0001)),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9.5,
            color=BLUE, fontweight="bold")

ax.set_xlabel("Epoch", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Val Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "SIL Training — Learning Rate Comparison, Second Day Triple\n"
    "GNN-free ScoringMLP, seed=42, 1/4 granularity (ss60/bi240), Train=05.01, Val=06.01",
    fontsize=10.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_xticks(epochs)
ax.set_ylim(0, 80)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")

plt.tight_layout()
out_pdf = "figures_export/nyc_sil_newdays_lr_comparison.pdf"
out_png = "figures_export/nyc_sil_newdays_lr_comparison.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
