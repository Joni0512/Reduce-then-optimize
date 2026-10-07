"""2026-10-05: bar-chart version of plot_rho_step_vs_batch_sweep.py - one panel per step_size, bars per batch_interval.
RHO, card 2, KEEP_ACTIVE=True, 9 VAL_INSTANCES. Same data sources/run counts as the line chart (see that docstring).
Hatched bar = single run (seed 42); solid bar = mean of several runs (n in label)."""
import re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
import json
# 2026-10-05: sweep points now from per-run summary.json (runs 1 and 2) instead of sweep.log, n = number of runs found
def sweep_runs(ss, bi):
    """2026-10-05: per-instance aggregation so aborted instances don't inflate a run's mean.
    Returns (mean, std, n): mean = mean over instances of the per-instance mean over successful runs;
    std = std of run means restricted to instances that succeeded in ALL runs of this config."""
    fs = sorted(glob.glob(str(root / f"outputs/rho_val_runs/bi{bi}_ss{ss}_card2_keepactiveTrue_seed42/run*/summary.json")))
    runs = [json.load(open(f))["rates"] for f in fs]
    insts = sorted(set().union(*runs))
    mean = float(np.mean([np.mean([r[i] for r in runs if i in r]) for i in insts]))
    common = [i for i in insts if all(i in r for r in runs)]
    rm = [np.mean([r[i] for i in common]) for r in runs]
    return mean, (float(np.std(rm)) if len(rm) > 1 else 0.0), len(runs)
def runmean(f):
    v = [float(m.group(1)) for m in (re.search(r"\] \w+: ([0-9.]+) \(", l) for l in open(f)) if m]; return np.mean(v)
def ms(v): return (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, len(v))
ss10 = {b: [runmean(f) for f in sorted(glob.glob(str(root / f"rho_val_bi{b}_ss10_card2*.log")))] for b in (20, 40, 60, 80)}
data = {  # step_size -> {batch_interval: (mean, std, n)}
    5:   {b: sweep_runs(5, b) for b in (5, 10, 20, 40, 60, 80)},
    10:  {10: sweep_runs(10, 10), **{b: (np.mean(v), np.std(v), len(v)) for b, v in ss10.items()}},
    20:  {b: sweep_runs(20, b) for b in (20, 40, 60, 80)},
    # step_size=100 bi200/400/600: mean/std of the 3 runs in plot_rho_horizon.py (0.6781/0.6978/0.7001, 0.6692/0.6453/0.6431, 0.6389/0.6254/0.6190)
    100: {100: sweep_runs(100, 100), 200: (0.692, 0.010, 3), 400: (0.653, 0.012, 3), 600: (0.628, 0.008, 3), 800: (0.606, 0, 1)},
}
col = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a", 100: "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "hatch.color": "white"})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True, gridspec_kw={"width_ratios": [6, 5, 4, 5]})
for ax, (ss, d) in zip(axes, data.items()):
    bis = list(d); x = np.arange(len(bis)); best = max(d[b][0] for b in bis)
    for xi, b in zip(x, bis):
        m, s, n = d[b]
        ax.bar(xi, m, 0.7, color=col[ss], hatch=None if n > 1 else "///", edgecolor="white", lw=0.5, zorder=2)
        if s: ax.errorbar(xi, m, yerr=s, color=INK, capsize=4, lw=1, zorder=3)
        ax.text(xi, m + s + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold" if m == best else None)
        if n > 1: ax.text(xi, 0.555, f"n={n}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax.set_xticks(x, [str(b) for b in bis]); ax.set_xlabel("batch_interval")
    ax.set_title(f"step_size={ss}", loc="left", fontsize=11, color=INK)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.55, 0.91); axes[0].set_ylabel("Service rate (mean over 9 val instances)")
# 2026-10-07: shorter footer, run setup (seed etc.) moved into a subtitle under the title (user request)
fig.suptitle("RHO service rate by step_size and batch_interval", x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.915, "cardinality 2, KEEP_ACTIVE=True, 9 val instances, seed 42; step_size=10 with batch_interval 20-80: 5 repeats + seeds 1-5 (n=10)",
         fontsize=8.5, color=INK2)
fig.text(0.01, 0.005, "Bar = mean over instances (failed instances excluded); error bar = std of run means; n = runs. "
         "Sources: outputs/rho_val_runs/*/run*/summary.json, rho_val_bi*_ss10_card2*.log, plot_rho_horizon.py (step_size=100).", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_step_vs_batch_sweep_card2_bars.{ext}", dpi=200)
