import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# label, card3_rate, card2_rate, card3_rt, card2_rt
rows = [
    ("8min/16min", 95.20, 96.70, 72, 62),
    ("4min/8min", 78.90, 77.70, 107, 97),
    ("4min/16min", 95.90, 97.10, 176, 125),
]

labels = [r[0] for r in rows]
card3_rate = [r[1] for r in rows]
card2_rate = [r[2] for r in rows]
card3_rt = [r[3] for r in rows]
card2_rt = [r[4] for r in rows]

x = range(len(labels))
width = 0.32

# ---- Service Rate ----
fig, ax = plt.subplots(figsize=(8.5, 5.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars3 = ax.bar([i - width / 2 for i in x], card3_rate, width, label="Cardinality 3", color=BLUE, zorder=3)
bars2 = ax.bar([i + width / 2 for i in x], card2_rate, width, label="Cardinality 2", color=ORANGE, zorder=3)

for bars in [bars3, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO — Cardinality 3 vs. 2 at Small/Medium Horizons: Service Rate\n"
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
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_cardinality2_vs_3_service_rate.pdf"
out_png = "figures_export/nyc_rho_cardinality2_vs_3_service_rate.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
plt.close(fig)

# ---- Runtime ----
fig, ax = plt.subplots(figsize=(8.5, 5.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars3 = ax.bar([i - width / 2 for i in x], card3_rt, width, label="Cardinality 3", color=BLUE, zorder=3)
bars2 = ax.bar([i + width / 2 for i in x], card2_rt, width, label="Cardinality 2", color=ORANGE, zorder=3)

for bars in [bars3, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}s", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Runtime (s)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO — Cardinality 3 vs. 2 at Small/Medium Horizons: Runtime\n"
    "Train day (12.01), seed 42, 1,000-request scale, 50 vehicles, no pruner",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)
ax.set_ylim(0, 200)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_cardinality2_vs_3_runtime.pdf"
out_png = "figures_export/nyc_rho_cardinality2_vs_3_runtime.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
