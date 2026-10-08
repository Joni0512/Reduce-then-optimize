"""2026-10-08: urgency-penalty test for the RHO horizon effect. Requests whose latest pickup is before
current_time + step_size get rejection penalty x factor (Config.URGENT_PENALTY_FACTOR, via Request.priority).
RHO offline, step_size=100, batch_interval 200-800, card 2, seed 42, 9 VAL_INSTANCES, runs 1-3 per config.
Sources: outputs/rho_val_runs/bi{bi}_ss100_card2_keepactive{True,False}_seed42[_urgent{2,4}]/run{1,2,3}/summary.json
(rtv_solver/pipeline/rho_val_horizon_runs.py --urgent_penalty_factor, drivers outputs/urgency_penalty/run_*.sh)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
BIS, FACTORS, KAS = [100, 200, 400, 600, 800], [1, 2, 4], [True, False]  # 2026-10-08: batch_interval=100 added
res = {}
print("keep_active factor batch_interval n_runs mean std min max failed")
for ka in KAS:
    for f in FACTORS:
        for bi in BIS:
            name = f"bi{bi}_ss100_card2_keepactive{ka}_seed42" + (f"_urgent{f}" if f != 1 else "")
            # 2026-10-08: all available runs (baseline batch_interval=100/KEEP_ACTIVE=True has 5, everything else 3)
            runs = [json.load(open(p)) for p in sorted((root / "outputs/rho_val_runs" / name).glob("run*/summary.json"))]
            m = [r["mean"] for r in runs]; fl = sum(len(r["failed"]) for r in runs)
            res[(ka, f, bi)] = (float(np.mean(m)), float(np.std(m)))
            print(ka, f, bi, len(m), round(np.mean(m), 4), round(np.std(m), 4), round(min(m), 4), round(max(m), 4), fl)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {1: "#8a8984", 2: "#2a78d6", 4: "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
w = 0.27; x = np.arange(len(BIS))
for ax, ka in zip(axes, KAS):
    for k, f in enumerate(FACTORS):
        mv = [res[(ka, f, b)][0] for b in BIS]; sv = [res[(ka, f, b)][1] for b in BIS]
        ax.bar(x + (k - 1) * w, mv, w, yerr=sv, capsize=2, color=COL[f], zorder=2,
               label="factor 1 (baseline)" if f == 1 else f"urgent rejection x{f}")
        for xi, v in zip(x + (k - 1) * w, mv):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, rotation=90, color=INK)
    ax.set_xticks(x, [str(b) for b in BIS]); ax.set_xlabel("batch_interval (step_size=100)", fontsize=9)
    ax.set_title(f"KEEP_ACTIVE={ka}", loc="left", fontsize=10)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.55, 0.88); axes[0].set_ylabel("Service rate (mean over 9 val instances)", fontsize=9)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.9))
fig.suptitle("Urgency-weighted rejection penalty vs RHO horizon", x=0.01, y=0.985, ha="left", fontsize=12)
fig.text(0.01, 0.925, "urgent = latest pickup before t + step_size; card 2, seed 42, 9 val instances, 3 runs per bar (baseline batch_interval=100 / KEEP_ACTIVE=True: 5), error bar = std of run means; no failed instances",
         fontsize=8, color=INK2)
fig.text(0.01, 0.005, "Sources: outputs/rho_val_runs/bi*_ss100_card2_keepactive*_seed42[_urgent2|_urgent4]/run{1,2,3}/summary.json", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.84))
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_urgency_penalty_horizon.{ext}", dpi=200)

# 2026-10-08: compact variant for the SRL meeting deck (rows = KEEP_ACTIVE, fits the slide's chart box)
fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.4), sharex=True, sharey=True)
for ax, ka in zip(axes, KAS):
    for k, f in enumerate(FACTORS):
        mv = [res[(ka, f, b)][0] for b in BIS]
        ax.bar(x + (k - 1) * w, mv, w, color=COL[f], zorder=2, label="factor 1 (baseline)" if f == 1 else f"urgent rejection x{f}")
        for xi, v in zip(x + (k - 1) * w, mv):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_title(f"KEEP_ACTIVE={ka}", loc="left", fontsize=9); ax.set_ylim(0.55, 0.87); ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True); ax.set_ylabel("service rate", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[1].set_xticks(x, [str(b) for b in BIS]); axes[1].set_xlabel("batch_interval (step_size=100)", fontsize=8)
axes[0].legend(fontsize=7, loc="upper right", ncol=3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_urgency_penalty_horizon_slide.{ext}", dpi=220)
