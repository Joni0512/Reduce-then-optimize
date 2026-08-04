"""Fleet-size ablation chart (service rate vs. number of vehicles), RHO 2/8,
Day A (2016-01-12, 06:00-14:00, 1,000 requests). Single series line chart."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

vehicles = [10, 20, 30, 40, 50, 60]
service_rate = [28.4, 56.2, 71.2, 77.9, 79.7, 82.3]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    vehicles, service_rate,
    color=BLUE, linewidth=2, marker="o", markersize=8,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
    solid_capstyle="round", zorder=3,
)

# selective direct labels at each point
for x, y in zip(vehicles, service_rate):
    ax.annotate(
        f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 12),
        ha="center", fontsize=10.5, color=TEXT_PRIMARY,
    )

ax.set_xlabel("Number of Vehicles", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "Fleet-Size Ablation — Service Rate vs. Number of Vehicles\n"
    "RHO 2/8, Day A (2016-01-12, 06:00-14:00, 1,000 requests)",
    fontsize=12.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 95)
ax.set_xticks(vehicles)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

plt.tight_layout()
out_path = "figures_export/nyc_fleet_ablation_chart.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_path)
pdf_path = "figures_export/nyc_fleet_ablation_chart.pdf"
plt.savefig(pdf_path, facecolor=fig.get_facecolor())
print("saved:", pdf_path)
