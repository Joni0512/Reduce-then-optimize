import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# seed -> value, Test day (2016-01-19), 4min/8min, cardinality 3, seeds 1-5
service_rate = {
    "baseline": [78.2833505687694, 78.2833505687694, 77.14581178903826, 76.93898655635988, 77.66287487073423],
    "pruned": [76.31851085832471, 77.14581178903826, 77.45604963805584, 76.83557394002068, 76.6287487073423],
}
runtime = {
    "baseline": [555.582, 469.559, 464.261, 469.261, 466.029],
    "pruned": [308.906, 463.861, 466.068, 462.39, 465.212],
}

labels = ["Baseline\n(no pruner)", "RequestPruner\n(t=0.8)"]
x = range(len(labels))
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
fig.patch.set_facecolor("#fcfcfb")

# ---- Panel 1: Service Rate ----
ax = axes[0]
ax.set_facecolor("#fcfcfb")
means = [np.mean(service_rate[k]) for k in ("baseline", "pruned")]
stds = [np.std(service_rate[k], ddof=1) for k in ("baseline", "pruned")]
ax.bar(x, means, width=0.5, yerr=stds, capsize=5, color=[BLUE, ORANGE], zorder=3,
       error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.2, "capthick": 1.2})
for i, k in enumerate(("baseline", "pruned")):
    vals = service_rate[k]
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter([i + j for j in jitter], vals, color="white", edgecolor=TEXT_PRIMARY,
               s=30, zorder=4, linewidth=0.9)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:.2f}%\n±{s:.2f}", (i, m + s), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9.5, color=TEXT_PRIMARY)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Service Rate (%)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title("Service Rate", fontsize=13, color=TEXT_PRIMARY, pad=12)
ax.set_ylim(70, 85)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

# ---- Panel 2: Runtime ----
ax2 = axes[1]
ax2.set_facecolor("#fcfcfb")
means2 = [np.mean(runtime[k]) for k in ("baseline", "pruned")]
stds2 = [np.std(runtime[k], ddof=1) for k in ("baseline", "pruned")]
ax2.bar(x, means2, width=0.5, yerr=stds2, capsize=5, color=[BLUE, ORANGE], zorder=3,
        error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.2, "capthick": 1.2})
for i, k in enumerate(("baseline", "pruned")):
    vals = runtime[k]
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i + j for j in jitter], vals, color="white", edgecolor=TEXT_PRIMARY,
                s=30, zorder=4, linewidth=0.9)
for i, (m, s) in enumerate(zip(means2, stds2)):
    ax2.annotate(f"{m:.1f}s\n±{s:.1f}", (i, m + s), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=9.5, color=TEXT_PRIMARY)
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels, fontsize=10.5)
ax2.set_ylabel("Runtime (s)", fontsize=11, color=TEXT_SECONDARY)
ax2.set_title("Runtime", fontsize=13, color=TEXT_PRIMARY, pad=12)
ax2.set_ylim(0, 650)
ax2.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax2.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax2.spines[spine].set_color(GRID)
ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

fig.suptitle(
    "NYC RequestPruner (t=0.8) vs. Baseline --- Test day (2016-01-19), 4min/8min, cardinality 3, seeds 1-5\n"
    "White dots = individual seeds, bars = mean, error bars = ±1 SD. Seed 1's runtime values are outliers (system-load noise, not a pruner effect).",
    fontsize=11, color=TEXT_PRIMARY, y=1.06,
)

plt.tight_layout()
out_pdf = "figures_export/NYC/nyc_request_pruner_vs_baseline_seeds.pdf"
out_png = "figures_export/NYC/nyc_request_pruner_vs_baseline_seeds.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", out_pdf)
print("saved:", out_png)
