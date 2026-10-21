"""2026-10-05: RHO service rate vs batch_interval for step_size 5/10/20/100 (card 2, KEEP_ACTIVE=True, 9 VAL_INSTANCES).
Sources (n = runs per point):
 ss5, ss20, ss10/bi10, ss100/bi100: outputs/rho_step_sweep/sweep.log (n=1, seed 42)
 ss10 bi20-80: rho_val_bi*_ss10_card2[_runN|_seedN].log (n=10: 5 repeats seed 42 + seeds 1-5), mean of run means
 ss100 bi200/400/600: n=3, bi800: n=1 (plot_rho_horizon.py values)."""
import re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
sweep = {}
for l in open(root / "outputs/rho_step_sweep/sweep.log"):
    m = re.search(r"bi=(\d+) ss=(\d+) .*MEAN over \d+ VAL_INSTANCES = ([0-9.]+)", l)
    if m: sweep[(int(m.group(2)), int(m.group(1)))] = float(m.group(3))
def runmean(f):
    v = [float(m.group(1)) for m in (re.search(r"\] \w+: ([0-9.]+) \(", l) for l in open(f)) if m]; return np.mean(v)
ss10 = {b: np.mean([runmean(f) for f in sorted(glob.glob(str(root / f"rho_val_bi{b}_ss10_card2*.log")))]) for b in (20, 40, 60, 80)}
ss10n = {b: len(glob.glob(str(root / f"rho_val_bi{b}_ss10_card2*.log"))) for b in (20, 40, 60, 80)}
series = {
    5:   ({b: sweep[(5, b)] for b in (5, 10, 20, 40, 60, 80)}, {b: 1 for b in (5, 10, 20, 40, 60, 80)}),
    10:  ({10: sweep[(10, 10)], **ss10}, {10: 1, **ss10n}),
    20:  ({b: sweep[(20, b)] for b in (20, 40, 60, 80)}, {b: 1 for b in (20, 40, 60, 80)}),
    100: ({100: sweep[(100, 100)], 200: 0.692, 400: 0.653, 600: 0.628, 800: 0.606}, {100: 1, 200: 3, 400: 3, 600: 3, 800: 1}),
}
col = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a", 100: "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(10, 5))
for ss, (vals, ns) in series.items():
    xs = sorted(vals); ys = [vals[x] for x in xs]
    ax.plot(xs, ys, color=col[ss], lw=2, zorder=3, label=f"step_size={ss}")
    for x, y in zip(xs, ys):
        ax.scatter(x, y, s=40 if ns[x] > 1 else 28, color=col[ss] if ns[x] > 1 else "white", edgecolor=col[ss], lw=1.5, zorder=4)
    ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
ax.set_xscale("log"); ticks = [5, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800]
ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
ax.set_xlabel("batch_interval (log scale)"); ax.set_ylabel("Service rate (mean over 9 val instances)"); ax.set_ylim(0.58, 0.9)
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("RHO: service rate vs. batch_interval for different step sizes", loc="left", fontsize=12, color=INK)
ax.scatter([], [], s=40, color=INK2, label="filled = mean of several runs"); ax.scatter([], [], s=28, color="white", edgecolor=INK2, label="hollow = single run")
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.text(0.01, 0.005, "RHO offline, MAX_CARDINALITY=2, KEEP_ACTIVE=True, 9 Li&Lim VAL_INSTANCES. Runs per point: step_size=10 batch_interval 20-80: 10 (5 repeats + 5 seeds); step_size=100 batch_interval 200/400/600: 3;\n"
         "all other points: 1 run (seed 42), solver noise ~±1-2 pp. Sources: outputs/rho_step_sweep/sweep.log, rho_val_bi*_ss10_card2*.log, plot_rho_horizon.py.", fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_step_vs_batch_sweep_card2.{ext}", dpi=200)
print({ss: {k: round(v, 4) for k, v in vals.items()} for ss, (vals, _) in series.items()})
