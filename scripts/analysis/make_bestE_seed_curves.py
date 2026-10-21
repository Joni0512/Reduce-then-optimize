"""
2026-10-10: validation service rate per epoch for all 5 seeds of config E (actor_lr 0.009315365931220558, critic_lr 0.0016399458556079478,
20 epochs) to show the fluctuations between seeds and over the epochs; thick line = mean over seeds with +-1 std band; dashed = RHO validation
(mean of 7 runs, keep_active=False, batch_interval=200/step_size=100). Seed 2079 = sweep trial ethereal-1 (bufsweep_local_44a8ed4b), seeds
101-104 = bestE_s*. PNG + PDF in figures_export/SRL/.
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
SW = REPO / "outputs" / "srl_training_sweep"
runs = [("seed 2079 (original)", SW / "bufsweep_local_44a8ed4b" / "result.json")] + [(f"seed {s}", SW / f"bestE_s{s}" / "result.json") for s in (101, 102, 103, 104)]
curves = {}
for lab, p in runs:
    d = json.load(open(p))
    curves[lab] = {r["epoch"]: r["service_rate"] for r in d["val_curve"]}
epochs = sorted(next(iter(curves.values())))
rho = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json"))) / 7

colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
fig, ax = plt.subplots(figsize=(10, 6))
for (lab, c), col in zip(curves.items(), colors):
    ax.plot(epochs, [c[e] for e in epochs], marker="o", linewidth=1.8, markersize=6, color=col, alpha=0.9, label=f"{lab}")
mean = [st.mean(c[e] for c in curves.values()) for e in epochs]
sd = [st.stdev(c[e] for c in curves.values()) for e in epochs]
ax.plot(epochs, mean, color="black", linewidth=3.2, marker="s", markersize=7, label="mean over 5 seeds")
ax.fill_between(epochs, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)], color="black", alpha=0.10, label="±1 std")
ax.axhline(rho, color="#444444", linestyle="--", linewidth=1.6, label=f"RHO validation, mean of 7 runs: {rho:.4f}")
for e, m, s in zip(epochs, mean, sd):
    ax.text(e + 0.25, m, f"{m:.3f}", fontsize=8, va="center")
ax.set_xticks(epochs); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation service rate (mean over 9 instances)")
ax.set_ylim(0.40, 0.80); ax.grid(True, alpha=0.3)
ax.set_title("Config E (actor_lr=0.009315, critic_lr=0.001640): validation per epoch for 5 seeds", fontsize=12)
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.text(0.01, 0.005, "Buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 pretrain + 20 epochs, validation every 5 epochs, "
         "batch_interval=200/step_size=100. Sources: result.json of the runs; RHO: outputs/rho_val_runs (keep_active=False).", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "bestE_5seeds_val_curves"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
for lab, c in curves.items():
    print(lab, {e: round(c[e], 3) for e in epochs})
print("mean", [round(m, 3) for m in mean], "std", [round(s, 3) for s in sd])
