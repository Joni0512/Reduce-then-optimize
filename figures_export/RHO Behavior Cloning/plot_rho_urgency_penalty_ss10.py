"""2026-10-08: urgency-penalty test at step_size=10 (batch_interval 20-80), same rule as plot_rho_urgency_penalty.py
(urgent = latest pickup before t + step_size, rejection penalty x factor). 9 VAL_INSTANCES, card 2, seed 42.
Sources: factor 2 (KEEP_ACTIVE True/False) and factor 1 KEEP_ACTIVE=False: outputs/rho_val_runs/bi{bi}_ss10_card2_keepactive*_seed42[_urgent2]/run{1,2,3}/summary.json;
factor 1 KEEP_ACTIVE=True: rho_val_bi{bi}_ss10_card2*.log in the repo root (5 repeats + 5 seeds, mean over successful instances per run)."""
import json, re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
BIS, FACTORS, KAS = [20, 40, 60, 80], [1, 2], [True, False]
def run_means(ka, f, bi):
    if ka and f == 1:
        out = []
        for p in sorted(glob.glob(str(root / f"rho_val_bi{bi}_ss10_card2*.log"))):
            v = [float(m[1]) for m in (re.search(r"\] \w+: ([0-9.]+) \(", l) for l in open(p)) if m]
            out.append(np.mean(v))
        return out
    name = f"bi{bi}_ss10_card2_keepactive{ka}_seed42" + (f"_urgent{f}" if f != 1 else "")
    return [json.load(open(p))["mean"] for p in sorted((root / "outputs/rho_val_runs" / name).glob("run*/summary.json"))]
res = {}
print("keep_active factor batch_interval n_runs mean std")
for ka in KAS:
    for f in FACTORS:
        for bi in BIS:
            m = run_means(ka, f, bi); res[(ka, f, bi)] = (float(np.mean(m)), float(np.std(m)), len(m))
            print(ka, f, bi, len(m), round(np.mean(m), 4), round(np.std(m), 4))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {1: "#8a8984", 2: "#2a78d6"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
w = 0.36; x = np.arange(len(BIS))
for ax, ka in zip(axes, KAS):
    for k, f in enumerate(FACTORS):
        mv = [res[(ka, f, b)][0] for b in BIS]; sv = [res[(ka, f, b)][1] for b in BIS]
        ax.bar(x + (k - 0.5) * w, mv, w, yerr=sv, capsize=2, color=COL[f], zorder=2,
               label="factor 1 (baseline)" if f == 1 else f"urgent rejection x{f}")
        for xi, v in zip(x + (k - 0.5) * w, mv):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [str(b) for b in BIS]); ax.set_xlabel("batch_interval (step_size=10)", fontsize=9)
    ax.set_title(f"KEEP_ACTIVE={ka}", loc="left", fontsize=10)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.80, 0.91); axes[0].set_ylabel("Service rate (mean over 9 val instances)", fontsize=9)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.9))
fig.suptitle("Urgency-weighted rejection penalty at step_size=10", x=0.01, y=0.985, ha="left", fontsize=12)
fig.text(0.01, 0.89, "urgent = latest pickup before t + 10; card 2, seed 42, 9 val instances; 3 runs per bar "
         "(baseline KEEP_ACTIVE=True: 10)\nerror bar = std of run means; y-axis starts at 0.80; one lc202 abort (factor 2, KEEP_ACTIVE=True, batch_interval=40, run 3)",
         fontsize=7.5, color=INK2)
fig.text(0.01, 0.005, "Sources: outputs/rho_val_runs/bi*_ss10_card2_keepactive*_seed42[_urgent2]/run*/summary.json, rho_val_bi*_ss10_card2*.log", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.84))
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_urgency_penalty_ss10.{ext}", dpi=200)

# 2026-10-08: compact variant for the SRL meeting deck (same panels, sized for the slide's chart box)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), sharey=True)
for ax, ka in zip(axes, KAS):
    for k, f in enumerate(FACTORS):
        mv = [res[(ka, f, b)][0] for b in BIS]; sv = [res[(ka, f, b)][1] for b in BIS]
        ax.bar(x + (k - 0.5) * w, mv, w, yerr=sv, capsize=2, color=COL[f], zorder=2,
               label="factor 1 (baseline)" if f == 1 else f"urgent rejection x{f}")
        for xi, v, s in zip(x + (k - 0.5) * w, mv, sv):
            ax.text(xi, v + s + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
    ax.set_xticks(x, [str(b) for b in BIS]); ax.set_xlabel("batch_interval (step_size=10)", fontsize=8)
    ax.set_title(f"KEEP_ACTIVE={ka}", loc="left", fontsize=9); ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.82, 0.925); axes[0].set_ylabel("service rate (y-axis from 0.82)", fontsize=8)
axes[0].legend(fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_urgency_penalty_ss10_slide.{ext}", dpi=220)
