"""GCN L1 pickup_slack ablation chart, v1 vs v2+competition, with vs without.
Grouped bar chart per seed, 4 bars per seed. SIL bi200/ss100 class1 legacy,
dropout=0.0, lr=0.001, hidden_dim=128."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

V1_COLOR = "#0b7a3f"
V1_PSLACK_COLOR = "#2a78d6"
V2_COLOR = "#e69f00"
V2_PSLACK_COLOR = "#b23a48"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

seeds = [1, 2, 3, 4, 5]
v1 = [0.7799, 0.7736, 0.7642, 0.7704, 0.7736]
v1_pslack = [0.7862, 0.7862, 0.7925, 0.7893, 0.7925]
v2 = [0.7516, 0.7704, 0.7421, 0.7547, 0.7390]
v2_pslack = [0.7673, 0.7610, 0.7736, 0.7673, 0.7642]

means = {
    "v1": (np.mean(v1), V1_COLOR),
    "v1+pickup_slack": (np.mean(v1_pslack), V1_PSLACK_COLOR),
    "v2+comp": (np.mean(v2), V2_COLOR),
    "v2+comp+pickup_slack": (np.mean(v2_pslack), V2_PSLACK_COLOR),
}

fig, ax = plt.subplots(figsize=(9.2, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(seeds))
w = 0.19
ax.bar(x - 1.5 * w, v1, w, color=V1_COLOR, edgecolor="#fcfcfb", linewidth=1, zorder=3, label="v1")
ax.bar(x - 0.5 * w, v1_pslack, w, color=V1_PSLACK_COLOR, edgecolor="#fcfcfb", linewidth=1, zorder=3, label="v1 + pickup_slack")
ax.bar(x + 0.5 * w, v2, w, color=V2_COLOR, edgecolor="#fcfcfb", linewidth=1, zorder=3, label="v2+comp")
ax.bar(x + 1.5 * w, v2_pslack, w, color=V2_PSLACK_COLOR, edgecolor="#fcfcfb", linewidth=1, zorder=3, label="v2+comp + pickup_slack")

ax.legend(loc="lower center", frameon=False, fontsize=9.5, labelcolor=TEXT_SECONDARY, ncol=2, bbox_to_anchor=(0.5, -0.28))

ax.set_xlabel("Seed", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Best Val Service Rate", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "GCN L1 — cr_pickup_slack Ablation on v1 vs. v2+Competition\n"
    "SIL bi200/ss100 class1 legacy, lr=0.001, hidden_dim=128, dropout=0.0",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0.70, 0.82)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seeds], fontsize=10.5, color=TEXT_SECONDARY)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
out_path = "figures_export/gcn_l1_pickup_slack_v1_vs_v2_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gcn_l1_pickup_slack_v1_vs_v2_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
