import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"
GREEN = "#3f9142"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

epochs = [1, 2, 3, 4, 5]
val_rate = [81.66, 93.31, 94.17, 90.45, 88.16]
best_epoch = 3
test_rate = 88.42

fig, ax = plt.subplots(figsize=(7.6, 4.9))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(
    epochs, val_rate, color=BLUE, linewidth=2, marker="o", markersize=8,
    markerfacecolor=BLUE, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
    zorder=3, label="Val (13.01), per epoch",
)
ax.scatter([best_epoch], [val_rate[best_epoch - 1]], s=220, facecolor="none",
           edgecolor=GREEN, linewidth=2.2, zorder=4)
ax.annotate(f"best epoch {best_epoch}\n{val_rate[best_epoch-1]:.2f}%",
            (best_epoch, val_rate[best_epoch - 1]), textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=10, color=GREEN, fontweight="bold")

ax.axhline(test_rate, color="#d67a2a", linewidth=1.8, linestyle="--", zorder=2)
ax.annotate(f"Test (19.01), ckpt ep.{best_epoch}: {test_rate:.2f}%",
            (1.05, test_rate), textcoords="offset points", xytext=(0, -14),
            ha="left", fontsize=10, color="#d67a2a", fontweight="bold")

ax.set_xlabel("Epoch", fontsize=11, color=TEXT_SECONDARY)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "SIL Training — 240 s Step Size / 16 min Horizon\n"
    "seed=42, LR=0.0001, Train=12.01, Val=13.01, Test=19.01",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_xticks(epochs)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")

plt.tight_layout()
out_pdf = "figures_export/nyc_sil_stepsize240_960.pdf"
out_png = "figures_export/nyc_sil_stepsize240_960.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
