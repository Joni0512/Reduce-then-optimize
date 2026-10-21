"""2026-10-01: RHO step_size=10, batch_interval=20/40/60/80, MAX_CARDINALITY=2, KEEP_ACTIVE=True, 9 VAL_INSTANCES.
(a) 5 repeated runs with seed 42: rho_val_bi{BI}_ss10_card2.log + ..._run{2..5}.log
(b) 5 different seeds 1-5:      rho_val_bi{BI}_ss10_card2_seed{1..5}.log
Run mean = mean over the instances that finished (infeasible-ILP aborts excluded)."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
bis = [20, 40, 60, 80]
def runmean(f):
    v = [float(m.group(1)) for m in (re.search(r"\] \w+: ([0-9.]+) \(", l) for l in open(f)) if m]
    return float(np.mean(v)), 9 - len(v)
rep = {b: [runmean(root / f"rho_val_bi{b}_ss10_card2.log")] + [runmean(root / f"rho_val_bi{b}_ss10_card2_run{r}.log") for r in range(2, 6)] for b in bis}
sed = {b: [runmean(root / f"rho_val_bi{b}_ss10_card2_seed{s}.log") for s in range(1, 6)] for b in bis}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; C = {"rep": "#2a78d6", "seed": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(9.5, 4.8)); x = np.arange(4); w = 0.36
for off, data, key, lab in [(-w/2, rep, "rep", "5 repeats, seed 42"), (w/2, sed, "seed", "5 seeds (1-5)")]:
    m = [np.mean([v for v, _ in data[b]]) for b in bis]; s = [np.std([v for v, _ in data[b]]) for b in bis]
    ax.bar(x + off, m, w - 0.04, color=C[key], label=lab, zorder=2)
    ax.errorbar(x + off, m, yerr=s, fmt="none", ecolor=INK, capsize=5, lw=1.1, zorder=4)
    for i, b in enumerate(bis):
        ax.scatter([x[i] + off] * 5, [v for v, _ in data[b]], s=22, color="white", edgecolor=INK, zorder=3)
        ax.text(x[i] + off, max(max(v for v, _ in data[b]), m[i] + s[i]) + 0.004, f"{m[i]:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    print(lab, {b: (round(mm, 4), round(ss, 4), sum(f for _, f in data[b])) for b, mm, ss in zip(bis, m, s)})
ax.set_xticks(x, [f"batch_interval={b}\nstep_size=10" for b in bis]); ax.set_ylim(0.80, 0.92)
ax.set_ylabel("Service rate (mean over 9 val instances)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("RHO step_size=10: repeated runs (seed 42) vs. different seeds", loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, loc="upper right", fontsize=9)
nf = lambda d: ", ".join(f"bi{b}: {sum(f for _, f in d[b])}" for b in bis)
fig.text(0.01, 0.005, "RHO offline, MAX_CARDINALITY=2, KEEP_ACTIVE=True, 9 Li&Lim VAL_INSTANCES. Bar = mean of 5 run means, error bar = std, dots = runs. Y-axis starts at 0.80.\n"
         f"Infeasible-ILP aborts (instance excluded from that run) - repeats: {nf(rep)}; seeds: {nf(sed)}. Sources: rho_val_bi*_ss10_card2[_runN|_seedN].log.", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_horizon_val_ss10_seeds_vs_repeats.{ext}", dpi=200)
