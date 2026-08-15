import numpy as np
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# seed -> (train, val, test) runtime (s)
data = {
    42: (39, 41, 35),
    1: (39, 41, 35),
    2: (38, 43, 37),
    3: (40, 44, 37),
    4: (41, 45, 40),
}

days = ["Train (12.01)", "Val (13.01)", "Test (19.01)"]
colors = ["#8fb8e8", BLUE, "#164a87"]

means = [np.mean([v[i] for v in data.values()]) for i in range(3)]
stds = [np.std([v[i] for v in data.values()], ddof=1) for i in range(3)]

fig, ax = plt.subplots(figsize=(8.2, 5.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = range(3)
bars = ax.bar(x, means, width=0.5, yerr=stds, capsize=6, color=colors, zorder=3,
              error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.3, "capthick": 1.3})

rng = np.random.default_rng(0)
for i, day_idx in enumerate(range(3)):
    vals = [v[day_idx] for v in data.values()]
    jitter = rng.uniform(-0.09, 0.09, size=len(vals))
    ax.scatter([i + j for j in jitter], vals, color="white", edgecolor=TEXT_PRIMARY,
               s=34, zorder=4, linewidth=1.0)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:.1f}s ± {s:.1f}", (i, m + s), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10, color=TEXT_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(days, fontsize=11)
ax.set_ylabel("Wall-clock Runtime (s)", fontsize=11, color=TEXT_SECONDARY)
ax.set_title(
    "RHO, Horizon = Step Size = 20 min (No Rolling Overlap) — 5-Seed Runtime Variance\n"
    "1,000-request scale, 50 vehicles, cardinality 3, original day triple, seeds 42/1/2/3/4",
    fontsize=12, color=TEXT_PRIMARY, pad=14,
)

ax.set_ylim(0, 55)
ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
ax.grid(axis="x", visible=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

plt.tight_layout()
out_pdf = "figures_export/nyc_rho_stepsize1200_seed_variance_runtime.pdf"
out_png = "figures_export/nyc_rho_stepsize1200_seed_variance_runtime.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor())
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
print("saved:", out_pdf)
print("saved:", out_png)
