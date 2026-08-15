import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# GNN v1, 2 message-passing layers, seed=42, NYC pair pruner, threshold=0.5 eval
pos_weights = [1, 2, 10, 20, 90]
f3_unbalanced = [0.000, 0.000, 0.048, 0.135, 0.1516]
f3_balanced = [0.1559, None, 0.1089, 0.1089, 0.1089]  # pw=2 balanced not tested

fig, ax = plt.subplots(figsize=(7.6, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    pos_weights, f3_unbalanced,
    color=BLUE, linewidth=2, marker="o", markersize=7,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="unbalanced",
)

bal_x = [pw for pw, v in zip(pos_weights, f3_balanced) if v is not None]
bal_y = [v for v in f3_balanced if v is not None]
ax.plot(
    bal_x, bal_y,
    color=ORANGE, linewidth=2, marker="s", markersize=7,
    markerfacecolor=ORANGE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="50:50 balanced",
)

for pw, v in zip(pos_weights, f3_unbalanced):
    y_offset = 10 if v < 0.02 else -16
    ax.annotate(f"{v:.3f}", (pw, v), textcoords="offset points", xytext=(0, y_offset),
                ha="center", fontsize=9, color=BLUE)
for pw, v in zip(bal_x, bal_y):
    ax.annotate(f"{v:.3f}", (pw, v), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=ORANGE)

ax.set_xscale("log")
ax.set_xticks(pos_weights)
ax.set_xticklabels([str(p) for p in pos_weights])
ax.set_xlabel("pos_weight", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("F3 Score (Val)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "NYC Pair Pruner — Imbalance-Handling Sweep\n"
    "GNN v1, 2 message-passing layers, seed=42, threshold=0.5",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 0.20)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper left")

plt.tight_layout()
out_pdf = "figures_export/nyc_pair_pruner_imbalance_sweep.pdf"
out_png = "figures_export/nyc_pair_pruner_imbalance_sweep.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
