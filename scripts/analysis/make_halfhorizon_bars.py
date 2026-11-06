"""
2026-10-29: bar chart SRL at half horizon (batch_interval=60/step_size=30, episodic scheme, 5 local seeds) vs RHO at batch_interval=120/step_size=30.
Bars = single seeds, dashed line = RHO mean of 3 runs, black line = SRL mean. Metrics: best val, val at epoch 20, test (checkpoint of best-val epoch).
Data: outputs/srl_training_sweep/hh_bi60ss30_allpre10ep_s*/result.json; RHO: rho_bi120_ss30_val.log / rho_bi120_ss30_test.log (3 runs each, "MEAN over 9 ..." lines):
val 0.7934/0.7956/0.7999 -> 0.7963, test 0.8145/0.8102/0.8102 -> 0.8116. PNG + PDF in figures_export/SRL/.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
runs = []
for d in sorted((REPO / "outputs" / "srl_training_sweep").glob("hh_bi60ss30_allpre10ep_s*")):
    j = json.load(open(d / "result.json"))
    curve = {pt["epoch"]: pt["service_rate"] for pt in j["val_curve"]}
    runs.append((j["seed"], j["best_val_service_rate"], curve[20], j["test_service_rate"]))
RHO = (0.7963, 0.7963, 0.8116)
names = ["best val", "val at epoch 20", "test"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for k, ax in enumerate(axes):
    vals = [r[k + 1] for r in runs]
    ax.bar(range(len(runs)), vals, color="#D55E00", alpha=0.8, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(RHO[k], color="gray", ls="--", lw=1.3, label=f"RHO bi120/ss30 {RHO[k]:.4f}")
    ax.axhline(st.mean(vals), color="black", lw=1.3, label=f"SRL mean {st.mean(vals):.4f}")
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([f"seed {r[0]}" for r in runs], fontsize=9)
    ax.set_title(names[k]); ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower right", fontsize=8, frameon=False)
axes[0].set_ylabel("service rate (pooled over the 9 validation / test instances)")
axes[0].set_ylim(0.6, 0.9)
fig.suptitle("SRL: batch_interval=60  step_size=30  actor_lr=0.0031923  critic_lr=0.002  epochs=20  episodic  reward=local_served  td_steps=1  batch 20/20  buffer=380 episodes\n"
             "pretrain=10 collected epochs (sigma 1.0) + 2000 critic steps  gamma=0.99  polyak_tau=0.005  sigma=0.2  num_samples=20  twin_critic  pickup_slack", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = REPO / "figures_export" / "SRL"
for ext in ("png", "pdf"):
    fig.savefig(out / f"halfhorizon_bi60ss30_vs_rho_bars.{ext}", dpi=160)
print(runs)
