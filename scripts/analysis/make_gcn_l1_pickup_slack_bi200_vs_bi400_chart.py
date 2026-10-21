"""GCN L1 pickup_slack ablation chart, batch_interval=200 vs. 400. Shows the
benefit is batch_interval-dependent, not general. SIL class1 legacy,
dropout=0.0, lr=0.001, hidden_dim=128, step_size=100 (both)."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

NO_PSLACK_COLOR = "#e69f00"
PSLACK_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

groups = ["batch_interval=200", "batch_interval=400"]
no_pslack_mean = [0.7723, 0.7560]
no_pslack_std = [0.0057, 0.0057]
pslack_mean = [0.7893, 0.7573]
pslack_std = [0.0032, 0.0084]

fig, ax = plt.subplots(figsize=(6.8, 5.0))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(groups))
w = 0.32
b1 = ax.bar(x - w / 2, no_pslack_mean, w, yerr=no_pslack_std, capsize=5, color=NO_PSLACK_COLOR, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="without pickup_slack")
b2 = ax.bar(x + w / 2, pslack_mean, w, yerr=pslack_std, capsize=5, color=PSLACK_COLOR, edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label="with pickup_slack")

for xi, v in zip(x, no_pslack_mean):
    ax.annotate(f"{v*100:.2f}%", (xi - w / 2, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=TEXT_PRIMARY)
for xi, v in zip(x, pslack_mean):
    ax.annotate(f"{v*100:.2f}%", (xi + w / 2, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=TEXT_PRIMARY)

ax.legend(loc="lower center", frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)

ax.set_ylabel("Best Val Service Rate (mean ± std, 5 seeds)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "GCN L1 — cr_pickup_slack Benefit Depends on batch_interval\n"
    "SIL class1 legacy, lr=0.001, hidden_dim=128, step_size=100",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0.70, 0.82)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11, color=TEXT_SECONDARY)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

plt.tight_layout()
out_path = "figures_export/gcn_l1_pickup_slack_bi200_vs_bi400_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gcn_l1_pickup_slack_bi200_vs_bi400_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
