"""
2026-10-12: validation service rate per epoch for the 30-epoch runs of config E (actor_lr 0.009315365931220558, critic_lr 0.0016399458556079478),
seeds 201-204 (seed 205 still running on the cluster, not included). Same layout as make_bestE_seed_curves.py (20 epochs, 5 other seeds).
Values copied from the cluster output of outputs/srl_training_sweep/bestE30_s<seed>/partial_val_curve.json (user paste, 2026-10-12); the runs'
result files are not local. Thick line = mean +-1 std; dashed = RHO validation (mean of 7 runs, keep_active=False, batch_interval=200/step_size=100).
PNG + PDF in figures_export/SRL/.
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
epochs = [5, 10, 15, 20, 25, 30]
curves = {
    "seed 201": [0.475, 0.519, 0.536, 0.549, 0.711, 0.642],
    "seed 202": [0.653, 0.696, 0.609, 0.633, 0.708, 0.643],
    "seed 203": [0.605, 0.596, 0.602, 0.680, 0.685, 0.635],
    "seed 204": [0.603, 0.639, 0.696, 0.592, 0.669, 0.644],
}
rho = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json"))) / 7

colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
fig, ax = plt.subplots(figsize=(10, 6))
for (lab, c), col in zip(curves.items(), colors):
    ax.plot(epochs, c, marker="o", linewidth=1.8, markersize=6, color=col, alpha=0.9, label=lab)
mean = [st.mean(c[i] for c in curves.values()) for i in range(len(epochs))]
sd = [st.stdev(c[i] for c in curves.values()) for i in range(len(epochs))]
ax.plot(epochs, mean, color="black", linewidth=3.2, marker="s", markersize=7, label="mean over 4 seeds")
ax.fill_between(epochs, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)], color="black", alpha=0.10, label="±1 std")
ax.axhline(rho, color="#444444", linestyle="--", linewidth=1.6, label=f"RHO validation, mean of 7 runs: {rho:.4f}")
for e, m in zip(epochs, mean):
    ax.text(e + 0.25, m, f"{m:.3f}", fontsize=8, va="center")
ax.set_xticks(epochs); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation service rate (mean over 9 instances)")
ax.set_ylim(0.40, 0.80); ax.grid(True, alpha=0.3)
ax.set_title("Config E (actor_lr=0.009315, critic_lr=0.001640), 30 epochs: validation per epoch for 4 seeds", fontsize=12)
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.text(0.01, 0.005, "Buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 pretrain + 30 epochs, validation every 5 epochs, batch_interval=200/step_size=100. "
         "Sources: bestE30_s201-204 partial_val_curve.json (cluster); RHO: outputs/rho_val_runs (keep_active=False). Seed 205 pending.", fontsize=6.3, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "bestE30_4seeds_val_curves"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("mean", [round(m, 3) for m in mean], "std", [round(s, 3) for s in sd], "rho", round(rho, 4))
