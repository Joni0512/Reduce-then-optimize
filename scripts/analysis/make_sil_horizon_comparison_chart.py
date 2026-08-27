import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

configs = ["4min/8min", "4min/16min", "8min/16min"]
best_val = [83.67, 96.66, 96.18]
test_rate = [75.90, 95.97, 94.21]

x = range(len(configs))
width = 0.32

fig, ax = plt.subplots(figsize=(9.0, 5.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars_val = ax.bar([i - width / 2 for i in x], best_val, width, label="Best Val (13.01)", color=BLUE, zorder=3)
bars_test = ax.bar([i + width / 2 for i in x], test_rate, width, label="Test (19.01)", color=ORANGE, zorder=3)

for bars in [bars_val, bars_test]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "SIL — Best Val vs. Test Service Rate by Step Size / Horizon\n"
    "Train = 2016-01-12, Val = 2016-01-13, Test = 2016-01-19 (seed 42, lr=0.0001, 5 epochs, cardinality 3)",
    fontsize=11.5, color=TEXT_PRIMARY, pad=14,
)
ax.set_ylim(0, 108)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper left")

plt.tight_layout()
out_pdf = "figures_export/nyc_sil_horizon_comparison.pdf"
out_png = "figures_export/nyc_sil_horizon_comparison.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
