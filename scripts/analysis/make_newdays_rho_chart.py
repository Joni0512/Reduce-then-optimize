import matplotlib.pyplot as plt

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

splits = ["Train\n05.01.2026", "Val\n06.01.2026", "Test\n20.01.2026"]
requests = [909, 953, 1081]
served = [630, 658, 721]
rates = [s / r * 100 for s, r in zip(served, requests)]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars = ax.bar(splits, rates, color=BLUE, width=0.55, zorder=3)

for bar, rate, s, r in zip(bars, rates, served, requests):
    ax.annotate(
        f"{rate:.2f}%\n({s}/{r})",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=10.5, color=TEXT_PRIMARY,
    )

ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO Baseline (No Learning) — New Day Triple\n"
    "1/4 granularity (step size 60s / batch interval 240s), 1,000-request scale, 50 vehicles",
    fontsize=11, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 85)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

plt.tight_layout()
out_pdf = "figures_export/nyc_newdays_rho_baseline.pdf"
out_png = "figures_export/nyc_newdays_rho_baseline.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
