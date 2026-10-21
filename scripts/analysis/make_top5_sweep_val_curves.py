"""
2026-10-10: validation service rate per epoch of the top-5 sweep trials (ranked by best_val_service_rate, the sweep metric, over the four
sweeps v1-v4 of the buffered-actor wandb project srl-buffered-sweep), see chat. Curves come from wandb (val_service_rate_curve logged by the
local agent / cluster coordinator after each trial); RHO reference = mean of the 7 repeated RHO validation runs (keep_active=False,
batch_interval=200/step_size=100). Writes PNG + PDF into figures_export/SRL/.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
ENTITY_PROJECT = "jonathan-homafar-technical-university-of-munich/srl-buffered-sweep"
SWEEPS = {"v1": "l1852691", "v2": "a209pqfi", "v3": "t9921npi", "v4": "k35e6qel"}
TOP_N = 5

api = wandb.Api()
cands = []
for name, sid in SWEEPS.items():
    for r in api.sweep(f"{ENTITY_PROJECT}/{sid}").runs:
        s = r.summary
        if s.get("best_val_service_rate") is not None and s.get("test_service_rate") is not None:
            pts = {int(x["epoch"]): x["val_service_rate_curve"] for x in r.scan_history(keys=["epoch", "val_service_rate_curve"]) if x.get("epoch")}
            cands.append(dict(sweep=name, name=r.name, alr=r.config["actor_lr"], clr=r.config.get("critic_lr"), seed=r.config["seed"], best=s["best_val_service_rate"],
                              test=s["test_service_rate"], curve=pts))
top = sorted(cands, key=lambda c: -c["best"])[:TOP_N]
rho = sum(json.load(open(p))["mean"] for p in sorted(glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json")))) / 7

colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
fig, ax = plt.subplots(figsize=(10, 6))
for c, col in zip(top, colors):
    xs = sorted(c["curve"])
    ax.plot(xs, [c["curve"][x] for x in xs], marker="o", linewidth=2.2, markersize=7, color=col,
            label=f"{c['name']} ({c['sweep']}): actor_lr={c['alr']:.5f}, critic_lr={c['clr']:.5f}, seed={c['seed']}  [best val {c['best']:.4f}, test {c['test']:.4f}]")
ax.axhline(rho, color="#444444", linestyle=":", linewidth=1.6, label=f"RHO validation, mean of 7 runs (batch_interval=200/step_size=100, keep_active=False): {rho:.4f}")
ax.set_xticks([5, 10, 15, 20]); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation service rate (mean, 9 instances)")
ax.set_ylim(0.5, 0.8); ax.grid(True, alpha=0.3)
ax.set_title("Top-5 sweep trials (by best validation service rate): validation service rate per epoch", fontsize=11)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
fig.text(0.01, 0.005, "Buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 pretrain + 20 epochs, n=1 per trial. Source: wandb project srl-buffered-sweep (val_service_rate_curve).",
         fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "top5_sweep_val_curves"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
for c in top: print(c["name"], {k: round(v, 3) for k, v in sorted(c["curve"].items())})
