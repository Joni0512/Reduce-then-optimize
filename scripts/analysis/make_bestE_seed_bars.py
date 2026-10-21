"""
2026-10-10: bar chart of the best-so-far config E (actor_lr 0.009315365931220558, critic_lr 0.0016399458556079478, 20 epochs) over 5 seeds:
best validation service rate over the epochs, validation service rate at epoch 20 and test service rate (best-val checkpoint) per seed, plus the mean +- std, against the RHO references
(Val 0.7043 = mean of 7 runs, Test 0.7379 = mean of 5 runs; keep_active=False, batch_interval=200/step_size=100). Seed 2079 is the original
sweep trial ethereal-1 (outputs/srl_training_sweep/bufsweep_local_44a8ed4b), seeds 101-104 the repeats (bestE_s*). PNG + PDF in figures_export/SRL/.
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
SW = REPO / "outputs" / "srl_training_sweep"
runs = [("2079\n(original)", SW / "bufsweep_local_44a8ed4b" / "result.json")] + [(str(s), SW / f"bestE_s{s}" / "result.json") for s in (101, 102, 103, 104)]
vals, tests, bests, labels = [], [], [], []
for lab, p in runs:
    d = json.load(open(p))
    vals.append({r["epoch"]: r["service_rate"] for r in d["val_curve"]}[20]); tests.append(d["test_service_rate"]); bests.append(d["best_val_service_rate"]); labels.append(lab)
rho_val = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json"))) / 7
rho_test = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run*/summary.json"))) / 5

x = np.arange(len(labels) + 1)
fig, ax = plt.subplots(figsize=(11.5, 6))
w = 0.26
series = [("Best validation epoch", bests, "#009E73", -w), ("Val @ epoch 20", vals, "#0072B2", 0.0), ("Test (best-val checkpoint)", tests, "#E69F00", w)]
for lab, data, col, off in series:
    ax.bar(x[:-1] + off, data, w, color=col, label=lab)
    ax.bar(x[-1] + off, st.mean(data), w, yerr=st.stdev(data), capsize=3, color=col, alpha=0.6, hatch="//", edgecolor="white")
    for xi, v in zip(x[:-1], data):
        ax.text(xi + off, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.text(x[-1] + off, st.mean(data) + st.stdev(data) + 0.005, f"{st.mean(data):.3f}", ha="center", fontsize=7)
ax.axhline(rho_val, color="#0072B2", linestyle="--", linewidth=1.5, label=f"RHO validation (7 runs): {rho_val:.4f}")
ax.axhline(rho_test, color="#E69F00", linestyle="--", linewidth=1.5, label=f"RHO test (5 runs): {rho_test:.4f}")
ax.set_xticks(x); ax.set_xticklabels(labels + ["mean ± std\n(5 seeds)"])
ax.set_xlabel("Seed"); ax.set_ylabel("Service rate (mean over 9 instances)"); ax.set_ylim(0.45, 0.85); ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Config E (actor_lr=0.009315, critic_lr=0.001640): five seeds vs. RHO", fontsize=12)
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.text(0.01, 0.005, "Buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 pretrain + 20 epochs, batch_interval=200/step_size=100. "
         "Sources: result.json of the runs; RHO: outputs/rho_val_runs (keep_active=False).", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "bestE_5seeds_bars"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("best", [round(v, 3) for v in bests], "mean", round(st.mean(bests), 3), "| val", [round(v, 3) for v in vals], "mean", round(st.mean(vals), 3), "| test", [round(t, 3) for t in tests], "mean", round(st.mean(tests), 3))
