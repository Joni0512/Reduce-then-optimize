import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# label (step/horizon in minutes), service_rate(%), runtime(s), cardinality
rows = [
    ("0.5min/8min", 81.40, 865, 3),
    ("1min/8min", 82.60, 406, 3),
    ("2min/8min\n(baseline)", 79.70, 209, 3),
    ("4min/8min", 78.90, 107, 3),
    ("4min/16min", 95.90, 176, 3),
    ("8min/16min", 95.20, 72, 3),
    ("10min/40min", 99.10, 961, 3),
    ("20min/20min\n(no overlap)", 95.50, 39, 3),
    ("30min/60min", 95.70, 8682, 3),
    ("60min/240min", 38.40, 11258, 2),
    ("120min/240min", 26.60, 5601, 2),
    ("120min/480min\n(full day)", 34.10, 6336, 2),
]

labels = [r[0] for r in rows]
rates = [r[1] for r in rows]
runtimes = [r[2] for r in rows]
colors = [BLUE if r[3] == 3 else ORANGE for r in rows]

x = range(len(labels))

# ---- Service Rate chart ----
fig, ax = plt.subplots(figsize=(15.0, 5.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars = ax.bar(x, rates, width=0.6, color=colors, zorder=3)
for bar, r in zip(bars, rates):
    ax.annotate(f"{r:.1f}%", (bar.get_x() + bar.get_width() / 2, r),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=9.5, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO (No Learning) — Full Horizon-Scaling Sweep: Service Rate\n"
    "Train day (12.01), seed 42, 1,000-request scale, 50 vehicles, no pruner",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)
ax.set_ylim(0, 108)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Cardinality 3"),
    plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Cardinality 2"),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left")

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_horizon_scaling_full_service_rate.pdf"
out_png = "figures_export/nyc_rho_horizon_scaling_full_service_rate.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
plt.close(fig)

# ---- Runtime chart (log scale) ----
fig, ax = plt.subplots(figsize=(15.0, 5.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars = ax.bar(x, runtimes, width=0.6, color=colors, zorder=3)
for bar, rt in zip(bars, runtimes):
    label = f"{rt}s" if rt < 3600 else f"{rt/3600:.1f}h"
    ax.annotate(label, (bar.get_x() + bar.get_width() / 2, rt),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=9.5, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_yscale("log")
ax.set_ylabel("Runtime (s, log scale)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO (No Learning) — Full Horizon-Scaling Sweep: Runtime\n"
    "Train day (12.01), seed 42, 1,000-request scale, 50 vehicles, no pruner",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)
ax.set_ylim(10, 20000)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0, which="both")
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

ax.legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left")

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_horizon_scaling_full_runtime.pdf"
out_png = "figures_export/nyc_rho_horizon_scaling_full_runtime.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
