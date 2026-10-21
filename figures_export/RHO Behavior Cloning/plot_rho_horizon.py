import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__import__("sys").argv[1]); out.mkdir(parents=True, exist_ok=True)
inst = ["lc103","lc202","lr103","lr107","lr202","lr208","lrc102","lrc202","lrc205"]
# per-instance service rates, 3 runs each (sources: see caption / chat 2026-09-27/28)
runs = {
 "batch_interval=200/step_size=100": [
  [0.8462,0.7647,0.7500,0.7308,0.5000,0.5400,0.7358,0.6275,0.6078],   # run A (compute_rho_val logic, 2026-09-27)
  [0.8462,0.9020,0.7500,0.7115,0.4800,0.5800,0.7358,0.6275,0.6471],   # run B (other session, 1. Lauf)
  [0.8462,0.8235,0.7500,0.7115,0.5400,0.5800,0.7358,0.6471,0.6667]],  # run C (other session, 2. Lauf)
 "batch_interval=400/step_size=100": [
  [0.8462,0.8431,0.7500,0.7308,0.4800,0.5000,0.7358,0.4706,0.6667],
  [0.8269,0.7843,0.7500,0.7308,0.4600,0.5000,0.7358,0.4118,0.6078],
  [0.8269,0.7647,0.7500,0.7308,0.4600,0.4800,0.7358,0.4510,0.5882]],
 "batch_interval=600/step_size=100": [
  [0.8269,0.7255,0.7500,0.7115,0.4200,0.6000,0.7358,0.3922,0.5882],
  [0.8269,0.7451,0.7500,0.7308,0.3400,0.5200,0.7358,0.3922,0.5882],
  [0.8077,0.7255,0.7500,0.7308,0.3800,0.5000,0.7358,0.3725,0.5686]],
 # 2026-09-28: single run by user (rho_val_bi800_ss100.log), MAX_CARDINALITY=2 explicit
 "batch_interval=800/step_size=100": [
  [0.7885,0.7059,0.7500,0.7308,0.4000,0.4200,0.7358,0.3725,0.5490]],
}
colors = ["#2a78d6","#eb6834","#1baf7a","#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2})
labels = list(runs)
means_run = {k: [np.mean(r) for r in v] for k,v in runs.items()}
cap = ("RHO (offline), Li&Lim VAL_INSTANCES (9), MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42; 3 runs each (batch_interval=800: 1 run) "
       "(solver non-deterministic: apply_async trip-cost order). batch_interval=200 runs 2-3 from a separate session.")

# Fig 1: mean over instances, bar = mean of 3 runs, dots = individual runs
fig, ax = plt.subplots(figsize=(8.6,4.4))
x = np.arange(len(labels))
for i,k in enumerate(labels):
    m = np.mean(means_run[k])
    ax.bar(x[i], m, width=0.55, color=colors[i], zorder=2)
    ax.scatter([x[i]]*len(means_run[k]), means_run[k], s=36, color="white", edgecolor=INK, linewidth=1, zorder=3)
    ax.text(x[i], max(means_run[k])+0.01, f"{m:.3f}", ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
ax.set_xticks(x, [l.replace("/", "\n") for l in labels])
ax.set_ylim(0.5, 0.75); ax.set_ylabel("Service rate (mean over 9 val instances)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.set_title("RHO horizon comparison on validation instances", loc="left", color=INK, fontsize=12)
ax.scatter([],[], s=36, color="white", edgecolor=INK, label="single run"); ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.005, cap + " Y-axis starts at 0.5.", fontsize=7, color=INK2, wrap=True)
fig.tight_layout(rect=(0,0.06,1,1))
for ext in ("png","pdf"): fig.savefig(out/f"rho_horizon_val_mean_bi200_bi400_bi600_bi800_ss100.{ext}", dpi=200)

# Fig 2: per instance, grouped, bar = mean of 3 runs, whisker = min..max
fig, ax = plt.subplots(figsize=(11,4.8))
w = 0.2; xi = np.arange(len(inst))
for i,k in enumerate(labels):
    arr = np.array(runs[k]); m = arr.mean(0)
    pos = xi + (i-1.5)*w
    ax.bar(pos, m, width=w-0.02, color=colors[i], label=k, zorder=2)
    ax.vlines(pos, arr.min(0), arr.max(0), color=INK, linewidth=1, zorder=3)
ax.set_xticks(xi, inst); ax.set_ylim(0,1); ax.set_ylabel("Service rate (mean of runs; line = min–max)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.set_title("RHO horizon comparison per validation instance", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
fig.text(0.01, 0.005, cap.replace(" (solver", "\n(solver"), fontsize=7, color=INK2)
fig.tight_layout(rect=(0,0.08,1,1))
for ext in ("png","pdf"): fig.savefig(out/f"rho_horizon_val_per_instance_bi200_bi400_bi600_bi800_ss100.{ext}", dpi=200)
for k in labels: print(k, [round(v,4) for v in means_run[k]], round(np.mean(means_run[k]),4))
