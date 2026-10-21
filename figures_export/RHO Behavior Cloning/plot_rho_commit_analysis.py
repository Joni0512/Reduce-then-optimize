"""RHO commit analysis (card=2, step_size=100, KEEP_ACTIVE=True, seed=42, 9 VAL_INSTANCES, 1 run per batch_interval).
Source: outputs/commit_analysis/summary_bi{200,400,600,800}_card2.json (rho_commit_leadtime.py, 2026-09-28)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
bis = [200, 400, 600, 800]
S = {b: json.load(open(root/"outputs"/"commit_analysis"/f"summary_bi{b}_card2.json")) for b in bis}
col = {200:"#2a78d6", 400:"#eb6834", 600:"#1baf7a", 800:"#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK2,"xtick.color":INK2,"ytick.color":INK2})
panels = [("service_rate","Service rate",None), ("frac_slots_full","Share of vehicle-steps with both\ntrip slots taken by commitments",None),
          ("pending_mean","Committed, not yet picked up\nrequests per vehicle (mean)",None), ("lead_mean","Mean lead time commit → pickup [s]",None)]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax,(k,t,_) in zip(axes, panels):
    v = [S[b]["mean"][k] for b in bis]
    ax.bar(range(4), v, color=[col[b] for b in bis], width=0.65, zorder=2)
    for i,y in enumerate(v): ax.text(i, y*1.01, f"{y:.2f}" if k!="lead_mean" else f"{y:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(4), [f"batch_interval\n={b}" for b in bis]); ax.set_title(t, loc="left", fontsize=9.5, color=INK)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig.text(0.01,0.01,"RHO offline, batch_interval=200/400/600/800, step_size=100, MAX_CARDINALITY=2, KEEP_ACTIVE=True, seed=42; mean over 9 Li&Lim VAL_INSTANCES, 1 run each. "
         "Decision time = step at which the request first appears in a manifest (100 s resolution).", fontsize=7, color=INK2)
fig.tight_layout(rect=(0,0.05,1,1))
for ext in ("png","pdf"): fig.savefig(here/f"rho_commit_analysis_card2_bi200_bi400_bi600_bi800_ss100.{ext}", dpi=200)
