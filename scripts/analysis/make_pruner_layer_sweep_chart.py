import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

layers = [1, 2, 3]
f3_pw90_unbalanced = [0.1462, 0.1516, 0.1536]
f3_pw1_balanced = [0.1458, 0.1559, 0.1513]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    layers, f3_pw90_unbalanced,
    color=BLUE, linewidth=2, marker="o", markersize=7,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="pos_weight=90, unbalanced",
)
ax.plot(
    layers, f3_pw1_balanced,
    color=ORANGE, linewidth=2, marker="s", markersize=7,
    markerfacecolor=ORANGE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    zorder=3, label="pos_weight=1, 50:50 balanced",
)

for x, y in zip(layers, f3_pw90_unbalanced):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=9, color=BLUE)
for x, y in zip(layers, f3_pw1_balanced):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=ORANGE)

ax.set_xticks(layers)
ax.set_xlabel("Message-passing layers", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("F3 Score (Val)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "NYC Pair Pruner — Layer-Depth Sweep\n"
    "GNN v1, seed=42, threshold=0.5",
    fontsize=12.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0.10, 0.19)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=9.5, loc="upper center")

plt.tight_layout()
out_pdf = "figures_export/nyc_pair_pruner_layer_sweep.pdf"
out_png = "figures_export/nyc_pair_pruner_layer_sweep.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
