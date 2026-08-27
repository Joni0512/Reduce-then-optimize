import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

configs = ["4min/8min", "4min/16min", "8min/16min", "10min/40min", "30min/60min"]

# seed -> value, per config (seeds 1-4)
rate_data = {
    "4min/8min": [78.70, 78.50, 78.50, 78.30],
    "4min/16min": [96.70, 97.20, 97.00, 97.20],
    "8min/16min": [95.10, 94.90, 95.20, 95.10],
    "10min/40min": [99.10, 99.10, 99.10, 98.00],
    "30min/60min": [96.20, 96.20, 96.20, 96.00],
}
trip_data = {
    "4min/8min": [26.78, 26.68, 26.91, 26.85],
    "4min/16min": [29.69, 29.38, 29.55, 29.33],
    "8min/16min": [29.37, 29.33, 29.33, 29.18],
    "10min/40min": [36.39, 36.41, 37.81, 36.33],
    "30min/60min": [39.94, 39.84, 39.88, 39.77],
}

x = range(len(configs))
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
fig.patch.set_facecolor("#fcfcfb")

# ---- Panel 1: Service Rate ----
ax = axes[0]
ax.set_facecolor("#fcfcfb")
means = [np.mean(rate_data[c]) for c in configs]
stds = [np.std(rate_data[c], ddof=1) for c in configs]
bars = ax.bar(x, means, width=0.55, yerr=stds, capsize=5, color=BLUE, zorder=3,
              error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.2, "capthick": 1.2})
for i, c in enumerate(configs):
    vals = rate_data[c]
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter([i + j for j in jitter], vals, color="white", edgecolor=TEXT_PRIMARY,
               s=28, zorder=4, linewidth=0.9)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:.2f}%\n±{s:.2f}", (i, m + s), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title("Service Rate (4 seeds: 1,2,3,4)", fontsize=12.5, color=TEXT_PRIMARY, pad=12)
ax.set_ylim(70, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

# ---- Panel 2: Avg Trip Duration ----
ax2 = axes[1]
ax2.set_facecolor("#fcfcfb")
means2 = [np.mean(trip_data[c]) for c in configs]
stds2 = [np.std(trip_data[c], ddof=1) for c in configs]
bars2 = ax2.bar(x, means2, width=0.55, yerr=stds2, capsize=5, color=ORANGE, zorder=3,
                error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.2, "capthick": 1.2})
for i, c in enumerate(configs):
    vals = trip_data[c]
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i + j for j in jitter], vals, color="white", edgecolor=TEXT_PRIMARY,
                s=28, zorder=4, linewidth=0.9)
for i, (m, s) in enumerate(zip(means2, stds2)):
    ax2.annotate(f"{m:.2f}m\n±{s:.2f}", (i, m + s), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=9, color=TEXT_PRIMARY)

ax2.set_xticks(list(x))
ax2.set_xticklabels(configs, fontsize=10)
ax2.set_ylabel("Avg Trip Duration (min)", fontsize=11, color=TEXT_SECONDARY)
ax2.set_title("Avg Trip Duration (4 seeds: 1,2,3,4)", fontsize=12.5, color=TEXT_PRIMARY, pad=12)
ax2.set_ylim(0, 46)
ax2.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax2.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color(GRID)
ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

fig.suptitle(
    "RHO — Robustness Across Seeds (Train day 2016-01-12, cardinality 3, seeds 1-4)\n"
    "White dots = individual seeds, bars = mean, error bars = ±1 SD. Runtime omitted (dominated by system-load noise, not a seed effect).",
    fontsize=11.5, color=TEXT_PRIMARY, y=1.05,
)

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_robustness_seeds.pdf"
out_png = "figures_export/nyc_rho_robustness_seeds.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", out_pdf)
print("saved:", out_png)
