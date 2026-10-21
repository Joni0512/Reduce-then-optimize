"""GCN L1 pickup_slack ablation: bar chart of 5-seed service rates, with vs.
without cr_pickup_slack on top of feature_builder v2 (+ competition
features). SIL bi200/ss100 class1 legacy, dropout=0.0."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#e69f00"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"
V1_LINE = "#0b7a3f"

seeds = [1, 2, 3, 4, 5]
without = [0.7516, 0.7704, 0.7421, 0.7547, 0.7390]
with_ = [0.7673, 0.7610, 0.7736, 0.7673, 0.7642]
mean_without = np.mean(without)
std_without = np.std(without, ddof=1)
mean_with = np.mean(with_)
std_with = np.std(with_, ddof=1)
v1_baseline = 0.7723  # pure v1-feature grid-best 5-seed mean, for reference

fig, ax = plt.subplots(figsize=(7.6, 4.9))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(seeds))
w = 0.35
bars1 = ax.bar(x - w / 2, without, w, color=ORANGE, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="v2, without pickup_slack")
bars2 = ax.bar(x + w / 2, with_, w, color=BLUE, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="v2, with pickup_slack")

for xi, v in zip(x, without):
    ax.annotate(f"{v*100:.1f}%", (xi - w / 2, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=TEXT_PRIMARY)
for xi, v in zip(x, with_):
    ax.annotate(f"{v*100:.1f}%", (xi + w / 2, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=TEXT_PRIMARY)

ax.axhline(mean_without, color=ORANGE, linewidth=1.3, linestyle="--", zorder=2, label=f"Mean without = {mean_without*100:.2f}% (±{std_without*100:.2f}%)")
ax.axhline(mean_with, color=BLUE, linewidth=1.3, linestyle="--", zorder=2, label=f"Mean with = {mean_with*100:.2f}% (±{std_with*100:.2f}%)")
ax.axhline(v1_baseline, color=V1_LINE, linewidth=1.3, linestyle=":", zorder=2, label=f"v1 grid-best reference = {v1_baseline*100:.2f}%")

ax.legend(loc="lower center", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=1, bbox_to_anchor=(0.27, 0.0))

ax.set_xlabel("Seed", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Best Val Service Rate", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "GCN L1 — Effect of cr_pickup_slack on Feature Builder v2\n"
    "SIL bi200/ss100 class1 legacy, lr=0.001, hidden_dim=128, dropout=0.0",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0.65, 0.82)
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
out_path = "figures_export/gcn_l1_pickup_slack_ablation_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gcn_l1_pickup_slack_ablation_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
