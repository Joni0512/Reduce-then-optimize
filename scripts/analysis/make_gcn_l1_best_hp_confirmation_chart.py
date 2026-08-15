"""GCN L1 best-HP confirmation chart - lr=0.001, hidden_dim=128, 5 seeds,
compared against the MLP baseline. SIL bi200/ss100 class1 legacy."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"
MLP_LINE = "#e69f00"

seeds = [1, 2, 3, 4, 5]
values = [0.7799, 0.7736, 0.7642, 0.7704, 0.7736]
mean = np.mean(values)
std = np.std(values, ddof=1)
mlp_baseline = 0.7327  # MLP 5-seed mean from architecture screening

fig, ax = plt.subplots(figsize=(7.2, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(seeds))
bars = ax.bar(
    x, values, 0.55, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
)

for xi, v in zip(x, values):
    ax.annotate(
        f"{v*100:.2f}%", (xi, v), textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=10.5, color=TEXT_PRIMARY,
    )

# mean line + MLP baseline reference line, labelled via legend (bars are too
# close in height for inline text near the lines to avoid colliding with the
# per-bar value labels)
ax.axhline(
    mean, color=TEXT_PRIMARY, linewidth=1.5, linestyle="-", zorder=2,
    label=f"Mean = {mean*100:.2f}% (±{std*100:.2f}%)",
)
ax.axhline(
    mlp_baseline, color=MLP_LINE, linewidth=1.5, linestyle="--", zorder=2,
    label=f"MLP baseline = {mlp_baseline*100:.2f}%",
)
ax.legend(
    loc="lower center", frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY,
)

ax.set_xlabel("Seed", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Best Val Service Rate", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "GCN L1 Confirmation — lr=0.001, hidden_dim=128, 5 Seeds\n"
    "SIL bi200/ss100 class1 legacy, dropout=0.0 — best combo of the HP sweep",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 1.0)
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
out_path = "figures_export/gcn_l1_best_hp_confirmation_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/gcn_l1_best_hp_confirmation_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
