import matplotlib.pyplot as plt

ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# label, baseline runtime (s), pruned runtime (s)
rows = [
    ("RHO\nVal, seed42", 531, 574),
    ("RHO\nVal, seed1", 397, 393),
    ("RHO\nTest, seed1", 613, 543),
    ("RHO\nTest, seed42", 350, 351),
    ("SIL\nVal, seed42", 976, 954),
    ("SIL\nVal, seed1", 1213, 1124),
    ("SIL\nTest, seed1", 869, 976),
]

labels = [r[0] for r in rows]
baseline = [r[1] for r in rows]
pruned = [r[2] for r in rows]

x = range(len(labels))
width = 0.32

fig, ax = plt.subplots(figsize=(11.5, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars_base = ax.bar([i - width / 2 for i in x], baseline, width, label="Baseline (no pruner)", color="#164a87", zorder=3)
bars_pruned = ax.bar([i + width / 2 for i in x], pruned, width, label="With pair pruner", color=ORANGE, zorder=3)

for bars in [bars_base, bars_pruned]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}s", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Wall-clock Runtime (s)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "Pair Pruner in the Full Pipeline — Runtime\n"
    "GNN v1, pos_weight=90 unbalanced, 3 layers, threshold=0.5, ~68% avg. edge reduction",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 1400)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
out_pdf = "figures_export/nyc_pair_pruner_pipeline_runtime_comparison.pdf"
out_png = "figures_export/nyc_pair_pruner_pipeline_runtime_comparison.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
