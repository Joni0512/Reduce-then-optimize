import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
f3_pw1_balanced = [0.1213, 0.1255, 0.1336, 0.1435, 0.1559, 0.1562, 0.1335, 0.0999, 0.0515]
f3_pw90_unbalanced = [0.1248, 0.1302, 0.1373, 0.1439, 0.1516, 0.1552, 0.1389, 0.1082, 0.0528]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    thresholds, f3_pw1_balanced,
    color=BLUE, linewidth=2, marker="o", markersize=6,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    solid_capstyle="round", zorder=3, label="pos_weight=1, balanced",
)
ax.plot(
    thresholds, f3_pw90_unbalanced,
    color=ORANGE, linewidth=2, marker="s", markersize=6,
    markerfacecolor=ORANGE, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
    solid_capstyle="round", zorder=3, label="pos_weight=90, unbalanced",
)

# mark the peak of each curve
peak1_idx = f3_pw1_balanced.index(max(f3_pw1_balanced))
peak2_idx = f3_pw90_unbalanced.index(max(f3_pw90_unbalanced))
ax.annotate(
    f"peak {f3_pw1_balanced[peak1_idx]:.4f}",
    (thresholds[peak1_idx], f3_pw1_balanced[peak1_idx]),
    textcoords="offset points", xytext=(-10, -35),
    ha="center", fontsize=9.5, color=BLUE, fontweight="bold",
)
ax.annotate(
    f"peak {f3_pw90_unbalanced[peak2_idx]:.4f}",
    (thresholds[peak2_idx], f3_pw90_unbalanced[peak2_idx]),
    textcoords="offset points", xytext=(35, -55),
    ha="center", fontsize=9.5, color=ORANGE, fontweight="bold",
)

ax.set_xlabel("Threshold", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("F3 Score", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "NYC Pair Pruner — F3 vs. Threshold\n"
    "GNN v1, trained on Day A (12.01), evaluated on Day B (Val, 13.01)",
    fontsize=12.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, max(max(f3_pw1_balanced), max(f3_pw90_unbalanced)) * 1.25)
ax.set_xticks(thresholds)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
out_pdf = "figures_export/nyc_pair_pruner_threshold_sweep.pdf"
out_png = "figures_export/nyc_pair_pruner_threshold_sweep.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
