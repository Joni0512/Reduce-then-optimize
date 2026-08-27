import matplotlib.pyplot as plt

BLUE = "#164a87"
MID = "#4a90c9"
LIGHT = "#a9cbe8"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

configs = ["4min/8min", "4min/16min", "8min/16min"]

# trip-size distribution (count of trips with 1/2/3 requests)
size1 = [834, 1032, 481]
size2 = [434, 915, 447]
size3 = [100, 472, 244]

avg_req_per_trip = [1.46, 1.77, 1.80]
avg_trip_span = [26.78, 29.40, 29.55]  # minutes, actual isolated trip duration

x = range(len(configs))
width = 0.55

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
fig.patch.set_facecolor("#fcfcfb")

# ---- Chart 1: trip size distribution (stacked) ----
ax = axes[0]
ax.set_facecolor("#fcfcfb")
b1 = ax.bar(x, size1, width, label="1 request", color=LIGHT, zorder=3)
b2 = ax.bar(x, size2, width, bottom=size1, label="2 requests", color=MID, zorder=3)
bottom2 = [a + b for a, b in zip(size1, size2)]
b3 = ax.bar(x, size3, width, bottom=bottom2, label="3 requests", color=BLUE, zorder=3)

totals = [a + b + c for a, b, c in zip(size1, size2, size3)]
for i, t in enumerate(totals):
    ax.annotate(f"n={t}", (i, t), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=10, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel("Number of Selected Trips", fontsize=11, color=TEXT_SECONDARY)
ax.set_title("Trip-Size Distribution", fontsize=13, color=TEXT_PRIMARY, pad=12)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

# ---- Chart 2: actual trip duration by trip size (span vs. summed-if-separate) ----
ax2 = axes[1]
ax2.set_facecolor("#fcfcfb")

sizes_labels = ["1 req", "2 req", "3 req"]
# pooled averages across all 3 configs for the by-size breakdown
span_by_size = [24.20, 31.50, 35.57]
direct_by_size = [19.37, 39.36, 58.58]

xs = range(len(sizes_labels))
w = 0.35
bars_span = ax2.bar([i - w / 2 for i in xs], span_by_size, w,
                     label="Actual trip duration (span)", color=BLUE, zorder=3)
bars_direct = ax2.bar([i + w / 2 for i in xs], direct_by_size, w,
                       label="Sum of individual direct times", color=ORANGE, zorder=3)

for bars in [bars_span, bars_direct]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}m", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=10, color=TEXT_PRIMARY)

ax2.set_xticks(list(xs))
ax2.set_xticklabels(sizes_labels, fontsize=11)
ax2.set_ylabel("Minutes", fontsize=11, color=TEXT_SECONDARY)
ax2.set_ylim(0, 68)
ax2.set_title("Trip Duration by Trip Size (pooled across configs)", fontsize=13, color=TEXT_PRIMARY, pad=12)
ax2.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax2.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color(GRID)
ax2.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax2.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "RHO — Trip-Level Statistics (isolated per-trip logging, co_base.py)\n"
    "Train day (12.01), seed 42, cardinality 3, 1,000-request scale, 50 vehicles, no pruner",
    fontsize=12.5, color=TEXT_PRIMARY, y=1.04,
)

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_trip_request_stats.pdf"
out_png = "figures_export/nyc_rho_trip_request_stats.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", out_pdf)
print("saved:", out_png)
