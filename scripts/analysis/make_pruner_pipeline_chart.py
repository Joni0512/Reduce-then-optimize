import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# label, baseline SR, pruned SR
rows = [
    ("RHO\nVal, seed42", 73.73, 72.97),
    ("RHO\nVal, seed1", 73.26, 73.26),
    ("RHO\nTest, seed1", 68.98, 68.98),
    ("RHO\nTest, seed42", 69.18, 68.25),
    ("SIL\nVal, seed42", 72.11, 70.77),
    ("SIL\nVal, seed1", 63.32, 61.89),
    ("SIL\nTest, seed1", 59.88, 55.43),
]

labels = [r[0] for r in rows]
baseline = [r[1] for r in rows]
pruned = [r[2] for r in rows]

x = range(len(labels))
width = 0.32

fig, ax = plt.subplots(figsize=(11.5, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars_base = ax.bar([i - width/2 for i in x], baseline, width, label="Baseline (no pruner)", color="#164a87", zorder=3)
bars_pruned = ax.bar([i + width/2 for i in x], pruned, width, label="With pair pruner", color=ORANGE, zorder=3)

for bars in [bars_base, bars_pruned]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "Pair Pruner in the Full Pipeline — Service Rate\n"
    "GNN v1, pos_weight=90 unbalanced, 3 layers, threshold=0.5, ~68% avg. edge reduction",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 85)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
out_pdf = "figures_export/nyc_pair_pruner_pipeline_comparison.pdf"
out_png = "figures_export/nyc_pair_pruner_pipeline_comparison.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
