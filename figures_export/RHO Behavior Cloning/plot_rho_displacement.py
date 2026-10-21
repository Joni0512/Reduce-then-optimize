"""2026-09-30: 'last chance' displacement analysis, RHO step_size=100, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42,
9 VAL_INSTANCES. Counts from outputs/displacement/analyze_last_chance.py (steps_bi{200,800}.json)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent
cats = ["less urgent:\nonly committed", "less urgent:\ncommitted + new", "less urgent:\nonly new", "only equally /\nmore urgent"]
groups = {
    "Lost at batch_interval=800,\nserved at 200 (n=58)": [6, 29, 0, 23],
    "All unserved,\nbatch_interval=200 (n=140)": [15, 34, 47, 44],
    "All unserved,\nbatch_interval=800 (n=185)": [23, 72, 31, 59],
}
colors = ["#eb6834", "#eda100", "#2a78d6", "#8a8984"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(groups)); w = 0.2
for i, c in enumerate(cats):
    vals = [v[i] / sum(v) for v in groups.values()]
    pos = x + (i - 1.5) * w
    ax.bar(pos, vals, width=w - 0.02, color=colors[i], label=c.replace("\n", " "), zorder=2)
    for p, y, n in zip(pos, vals, [v[i] for v in groups.values()]):
        ax.text(p, y + 0.01, f"{n}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x, list(groups)); ax.set_ylabel("Share of requests"); ax.set_ylim(0, 0.75)
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("At a request's last chance: what did the vehicles that could have served it take?", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
fig.text(0.01, 0.01, "RHO offline, step_size=100, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42, 9 Li&Lim VAL_INSTANCES, 1 run each. Last chance = last step in which the request was rejected\n"
         "although a vehicle had a feasible candidate for it. 'Less urgent' = assigned request's slack (latest pickup - t) >= lost request's slack + 100 s. Bar labels = counts.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_displacement_last_chance_bi200_vs_bi800_ss100.{ext}", dpi=200)
