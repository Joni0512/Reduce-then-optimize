import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
MID = "#4a90c9"
LIGHT = "#a9cbe8"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

configs = ["4min/8min", "4min/16min", "8min/16min"]

# trip-size distribution (count of trips with 1/2/3 requests)
size1 = [858, 945, 521]
size2 = [444, 890, 431]
size3 = [88, 514, 246]

avg_req_per_trip = [1.45, 1.82, 1.77]
avg_direct_trip_time = [27.93, 35.74, 34.72]  # minutes

x = range(len(configs))
width = 0.55

# ---- Chart 1: trip size distribution (stacked) ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
fig.patch.set_facecolor("#fcfcfb")

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

# ---- Chart 2: avg requests/trip + avg direct trip time ----
ax2 = axes[1]
ax2.set_facecolor("#fcfcfb")
width2 = 0.35
bars_req = ax2.bar([i - width2 / 2 for i in x], avg_req_per_trip, width2,
                    label="Avg requests/trip", color=BLUE, zorder=3)
for bar, v in zip(bars_req, avg_req_per_trip):
    ax2.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                 textcoords="offset points", xytext=(0, 4), ha="center",
                 fontsize=10, color=TEXT_PRIMARY)
ax2.set_ylabel("Avg Requests per Trip", fontsize=11, color=BLUE)
ax2.set_ylim(0, 2.5)
ax2.set_xticks(list(x))
ax2.set_xticklabels(configs, fontsize=11)
ax2.tick_params(axis="y", colors=BLUE, labelsize=10)
for spine in ["top"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["left"].set_color(BLUE)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color(GRID)

ax3 = ax2.twinx()
bars_time = ax3.bar([i + width2 / 2 for i in x], avg_direct_trip_time, width2,
                     label="Avg direct trip time (min)", color="#d67a2a", zorder=3)
for bar, v in zip(bars_time, avg_direct_trip_time):
    ax3.annotate(f"{v:.1f}m", (bar.get_x() + bar.get_width() / 2, v),
                 textcoords="offset points", xytext=(0, 4), ha="center",
                 fontsize=10, color=TEXT_PRIMARY)
ax3.set_ylabel("Avg Direct Trip Travel Time (min)", fontsize=11, color="#d67a2a")
ax3.set_ylim(0, 42)
ax3.tick_params(axis="y", colors="#d67a2a", labelsize=10)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_color("#d67a2a")

ax2.set_title("Trip Composition & Direct Travel Time", fontsize=13, color=TEXT_PRIMARY, pad=12)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9.5, loc="upper left")

fig.suptitle(
    "RHO — Trip-Level Statistics (from new per-trip logging)\n"
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
