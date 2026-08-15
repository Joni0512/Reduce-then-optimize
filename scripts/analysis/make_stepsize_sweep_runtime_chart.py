import matplotlib.pyplot as plt

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

configs = ["120s / 8min\n(baseline)", "60s / 8min", "30s / 8min", "240s / 16min", "1200s / 20min\n(no overlap)"]
train_rt = [209, 406, 865, 176, 39]
val_rt = [225, 464, 913, 211, 41]
test_rt = [195, 427, 809, 166, 35]

x = range(len(configs))
width = 0.25

fig, ax = plt.subplots(figsize=(10.5, 5.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

bars_train = ax.bar([i - width for i in x], train_rt, width, label="Train (12.01)", color="#8fb8e8", zorder=3)
bars_val = ax.bar([i for i in x], val_rt, width, label="Val (13.01)", color=BLUE, zorder=3)
bars_test = ax.bar([i + width for i in x], test_rt, width, label="Test (19.01)", color="#164a87", zorder=3)

for bars in [bars_train, bars_val, bars_test]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}s", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.5, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel("Runtime (s)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO (No Learning) — Step Size / Horizon Sweep: Runtime\n"
    "1,000-request scale, 50 vehicles, cardinality 3, original day triple",
    fontsize=12.5, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 1050)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=3)

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_stepsize_sweep_runtime.pdf"
out_png = "figures_export/nyc_rho_stepsize_sweep_runtime.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
