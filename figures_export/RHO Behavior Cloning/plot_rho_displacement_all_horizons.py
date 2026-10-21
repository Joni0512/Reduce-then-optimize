"""2026-09-30: last-chance displacement for requests served at batch_interval=200 but lost at 400/600/800
(RHO step_size=100, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42, 9 VAL_INSTANCES, 1 run each).
Source: outputs/displacement/summary_all_horizons.json (analyze_last_chance_all.py)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
S = json.load(open(root / "outputs/displacement/summary_all_horizons.json"))
keys = [k for k in S if k.startswith("lost at")]
labels = [f"batch_interval={k.split('bi')[1].split(',')[0]}\n(n={S[k]['n']})" for k in keys]
cats = [("only committed", "#eb6834", "less urgent: only committed"), ("committed + new", "#eda100", "less urgent: committed + new"),
        ("only new", "#2a78d6", "less urgent: only new"), ("equal/more urgent", "#8a8984", "only equally / more urgent")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(keys)); bottom = np.zeros(len(keys))
for key, col, lab in cats:
    v = np.array([S[k].get(key, 0) / S[k]["n"] for k in keys])
    ax.bar(x, v, bottom=bottom, color=col, width=0.55, label=lab, zorder=2, edgecolor="white", linewidth=2)
    for xi, b, h, n in zip(x, bottom, v, [S[k].get(key, 0) for k in keys]):
        if h > 0.06: ax.text(xi, b + h / 2, f"{n}", ha="center", va="center", fontsize=9, color="white" if col != "#eda100" else INK)
    bottom += v
ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_ylabel("Share of extra-lost requests")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Requests served at batch_interval=200 but lost at longer horizons: what took their slot?", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.text(0.01, 0.01, "RHO offline, step_size=100, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42, 9 Li&Lim VAL_INSTANCES, 1 run per horizon. Last chance = last step the request was rejected\n"
         "although a vehicle had a feasible candidate; 'less urgent' = assigned request's slack >= lost request's slack + 100 s. Bar labels = counts.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_displacement_last_chance_all_horizons_ss100.{ext}", dpi=200)
