import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
GREEN = "#2E7D5B"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

epochs = [1, 2, 3, 4, 5]
val_rates = [76.22, 80.99, 78.89, 83.67, 79.94]
best_epoch = 4
test_rate = 75.90

fig, ax = plt.subplots(figsize=(9.5, 5.6))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

colors = [GREEN if e == best_epoch else BLUE for e in epochs]
bars = ax.bar([f"Val\nEpoch {e}" for e in epochs], val_rates, width=0.55, color=colors, zorder=3)
for bar, r, e in zip(bars, val_rates, epochs):
    label = f"{r:.2f}%" + (" (best)" if e == best_epoch else "")
    ax.annotate(label, (bar.get_x() + bar.get_width() / 2, r),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=10, color=TEXT_PRIMARY, fontweight="bold" if e == best_epoch else "normal")

bar_test = ax.bar(["Test\n(best-val ckpt)"], [test_rate], width=0.55, color=ORANGE, zorder=3)
ax.annotate(f"{test_rate:.2f}%", (bar_test[0].get_x() + bar_test[0].get_width() / 2, test_rate),
            textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=10, color=TEXT_PRIMARY, fontweight="bold")

ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "SIL — Step Size 4 min / Horizon 8 min\n"
    "Train = 2016-01-12, Val = 2016-01-13, Test = 2016-01-19 (seed 42, lr=0.0001, 5 epochs, cardinality 3)",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Val (per epoch)"),
    plt.Rectangle((0, 0), 1, 1, color=GREEN, label="Val (best epoch)"),
    plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Test (best-val checkpoint)"),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=9.5, loc="upper left")

plt.tight_layout()
out_pdf = "figures_export/nyc_sil_stepsize4min8min.pdf"
out_png = "figures_export/nyc_sil_stepsize4min8min.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
