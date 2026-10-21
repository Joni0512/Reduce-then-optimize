"""2026-10-01: RHO step_size=10, batch_interval=20/40/60/80, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42, 9 VAL_INSTANCES, 5 runs
(2026-10-01: extended from 2 to 5 runs; run 1 = rho_val_bi{BI}_ss10_card2.log, runs 2-5 = ..._run{N}.log).
Parses rho_val_bi{BI}_ss10_card2.log (run 1) and rho_val_bi{BI}_ss10_card2_run2.log (run 2) from the repo root.
Run 1 lr202 at batch_interval=60 failed (InfeasibleAssignmentError) -> only run 2 used for that cell."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
inst = ["lc103","lc202","lr103","lr107","lr202","lr208","lrc102","lrc202","lrc205"]
bis = [20, 40, 60, 80]
def parse(f):
    d = {}
    for line in open(f):
        m = re.search(r"\] (\w+): ([0-9.]+) \(", line)
        if m: d[m.group(1)] = float(m.group(2))
    return d
NRUNS = 5
runs = {b: [parse(root / f"rho_val_bi{b}_ss10_card2.log")] + [parse(root / f"rho_val_bi{b}_ss10_card2_run{r}.log") for r in range(2, NRUNS + 1)] for b in bis}
fails = {b: sum(len(inst) - len(r) for r in runs[b]) for b in bis}
std = {b: float(np.std(run_means_tmp)) for b, run_means_tmp in [(b, [np.mean([r[i] for i in inst if i in r]) for r in runs[b]]) for b in bis]}
run_means = {b: [np.mean([r[i] for i in inst if i in r]) for r in runs[b]] for b in bis}
inst_means = {b: [np.mean([r[i] for r in runs[b] if i in r]) for i in inst] for b in bis}
mean = {b: float(np.mean(inst_means[b])) for b in bis}
colors = {20: "#2a78d6", 40: "#eb6834", 60: "#1baf7a", 80: "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
cap = ("RHO offline, step_size=10, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42, 9 Li&Lim VAL_INSTANCES, 5 runs each (solver non-deterministic). Bar = mean of per-instance means,\n"
       f"error bar = std over run means. Infeasible-ILP aborts (instance excluded from that run): " + ", ".join(f"bi{b}: {fails[b]}" for b in bis) + ". Sources: rho_val_bi*_ss10_card2[_runN].log.")
fig, ax = plt.subplots(figsize=(8.6, 4.6)); x = np.arange(4)
for i, b in enumerate(bis):
    ax.bar(i, mean[b], width=0.55, color=colors[b], zorder=2)
    ax.errorbar(i, mean[b], yerr=std[b], color=INK, capsize=6, lw=1.2, zorder=4)
    ax.scatter([i] * len(run_means[b]), run_means[b], s=30, color="white", edgecolor=INK, zorder=3)
    ax.text(i, max(run_means[b] + [mean[b] + std[b]]) + 0.006, f"{mean[b]:.3f} ± {std[b]:.3f}", ha="center", va="bottom", fontweight="bold", color=INK)
ax.scatter([], [], s=36, color="white", edgecolor=INK, label="single run (run mean over its instances)")
ax.set_xticks(x, [f"batch_interval={b}\nstep_size=10" for b in bis]); ax.set_ylim(0.75, 0.92)
ax.set_ylabel("Service rate (mean over 9 val instances)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("RHO with step_size=10: horizon comparison (validation instances)", loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.text(0.01, 0.005, cap.replace("aborts (instance", "aborts\n(instance") + " Y-axis starts at 0.75.", fontsize=6.3, color=INK2)
fig.tight_layout(rect=(0, 0.11, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_horizon_val_mean_ss10_bi20_bi40_bi60_bi80.{ext}", dpi=200)
fig, ax = plt.subplots(figsize=(12, 4.8)); xi = np.arange(len(inst)); w = 0.2
for k, b in enumerate(bis):
    pos = xi + (k - 1.5) * w
    vals = [[r[i] for r in runs[b] if i in r] for i in inst]
    ax.bar(pos, inst_means[b], width=w - 0.02, color=colors[b], label=f"batch_interval={b}/step_size=10 (mean {mean[b]:.3f})", zorder=2)
    ax.vlines(pos, [min(v) for v in vals], [max(v) for v in vals], color=INK, lw=1, zorder=3)
ax.set_xticks(xi, inst); ax.set_ylim(0, 1.08); ax.set_ylabel("Service rate (mean of 5 runs; line = min-max)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("RHO with step_size=10 per validation instance", loc="left", fontsize=12, color=INK)
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right", bbox_to_anchor=(1, 1.03))
fig.text(0.01, 0.005, cap, fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_horizon_val_per_instance_ss10_bi20_bi40_bi60_bi80.{ext}", dpi=200)
for b in bis: print(b, [round(v, 4) for v in run_means[b]], "mean", round(mean[b], 4), "std", round(std[b], 4), "fails", fails[b])
