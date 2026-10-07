"""2026-10-06: per-instance RHO service rate vs batch_interval (step_size=10, card 2, KEEP_ACTIVE=True, seed 42),
with instance time scales (horizon, median pickup-window width) in the x labels.
Sources: outputs/window_analysis/instance_time_scales.py -> outputs/window_analysis/summary.json (time scales);
rates: outputs/rho_val_runs/bi{bi}_ss10_card2_keepactiveTrue_seed42/run*/summary.json or rho_val_bi*_ss10_card2*.log."""
import json, sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "outputs/window_analysis"))
import os; os.chdir(ROOT)
from instance_time_scales import per_inst  # noqa: E402  (prints its table on import)
S = json.load(open(ROOT / "outputs/window_analysis/summary.json"))["scales"]
inst = sorted(S, key=lambda i: S[i]["horizon"])
bis = [10, 20, 40, 60, 80]; tab = {b: per_inst(10, b) for b in bis}
fig, ax = plt.subplots(figsize=(13, 4.8)); w = 0.16; cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(bis)))
for k, b in enumerate(bis):
    ax.bar(np.arange(len(inst)) + (k - 2) * w, [tab[b][i] for i in inst], w, color=cols[k], label=f"batch_interval={b}")
ax.set_xticks(range(len(inst)), [f"{i}\nhorizon {S[i]['horizon']:.0f}\nwindow {S[i]['win']:.0f}" for i in inst])
ax.set_ylim(0.65, 1.0); ax.set_ylabel("service rate (mean over runs)")
ax.set_title("RHO per instance, step_size=10, cardinality 2 - sorted by instance horizon (Li&Lim time units)")
ax.legend(ncol=5, fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=.3); fig.tight_layout()
out = ROOT / "figures_export/RHO Behavior Cloning/rho_best_bi_vs_time_scales"
fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
