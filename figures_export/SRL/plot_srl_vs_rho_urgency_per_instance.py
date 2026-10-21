"""2026-10-13: SRL vs two RHO variants per validation instance (step_size=100, batch_interval=200, card 2, KEEP_ACTIVE=False).
- RHO factor 1: outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json (7 runs)
- RHO factor 2 (urgent = latest pickup before t + step_size, rejection penalty x2, Config.URGENT_PENALTY_FACTOR):
  outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_urgent2/run*/summary.json (3 runs)
- SRL (actor_lr=0.0031923396291543868, critic_lr=0.002, buffered actor batch 16, twin critic, no warmstart, pickup_slack,
  10 pretrain + 20 epochs, batch_interval=200/step_size=100), epoch 20: outputs/srl_training_sweep/bestD_s301-305/result.json
  (val_curve per_instance, 5 seeds; updated from 4 seeds on 2026-10-13 after seed 305 finished)."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
def rho(name):
    d = {}
    for f in sorted(glob.glob(str(root / "outputs/rho_val_runs" / name / "run*/summary.json"))):
        for k, v in json.load(open(f))["rates"].items():
            d.setdefault(k, []).append(v)
    return d
r1 = rho("bi200_ss100_card2_keepactiveFalse_seed42"); r2 = rho("bi200_ss100_card2_keepactiveFalse_seed42_urgent2")
srl = {}
for f in sorted(glob.glob(str(root / "outputs/srl_training_sweep/bestD_s30*/result.json"))):
    ep = [e for e in json.load(open(f))["val_curve"] if e.get("epoch") == 20][0]
    for k, v in ep["per_instance"].items():
        srl.setdefault(k, []).append(v)
INST = sorted(srl)
series = [("RHO (urgency-blind penalty, 7 runs)", r1, "#8a8984"), ("RHO urgent rejection x2 (3 runs)", r2, "#1baf7a"),
          ("SRL epoch 20 (5 seeds)", srl, "#2a78d6")]
print("instance " + " | ".join(s[0] for s in series))
for i in INST + ["mean"]:
    vals = [np.mean([np.mean(d[k]) for k in INST]) if i == "mean" else np.mean(d[i]) for _, d, _ in series]
    print(i, [round(v, 3) for v in vals])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
labels = INST + ["mean"]; x = np.arange(len(labels)); w = 0.27
fig, ax = plt.subplots(figsize=(13, 4.8))
for k, (lab, d, c) in enumerate(series):
    m = [np.mean(d[i]) for i in INST] + [np.mean([np.mean(d[i]) for i in INST])]
    s = [np.std(d[i]) for i in INST] + [0]
    xs = x + (k - 1) * w
    ax.bar(xs, m, w, yerr=s, capsize=2, color=c, label=lab, zorder=2)
    for xi, v, e in zip(xs, m, s):
        ax.text(xi, v + e + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
    if d is srl:
        for xi, i in zip(xs, INST):
            ax.scatter([xi] * len(d[i]), d[i], s=8, color=INK, zorder=3)
ax.axvline(len(INST) - 0.5, color=INK2, lw=0.8, ls=":")
ax.set_xticks(x, labels); ax.set_ylim(0.4, 1.05); ax.set_ylabel("service rate (validation)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
fig.suptitle("SRL vs RHO with and without urgency-weighted rejection penalty", x=0.01, y=0.99, ha="left", fontsize=12)
fig.text(0.01, 0.005, "All: step_size=100, batch_interval=200, card 2, KEEP_ACTIVE=False, 9 val instances. SRL: actor_lr=0.003192, critic_lr=0.002, buffered actor (batch 16), "
         "twin critic, no warmstart, pickup_slack,\n10 pretrain + 20 epochs; dots = seeds 301-305. Error bars = std over runs/seeds. "
         "Sources: outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42[_urgent2]/run*/summary.json, outputs/srl_training_sweep/bestD_s301-305/result.json",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(here / f"srl_vs_rho_urgency_per_instance.{ext}", dpi=200)
