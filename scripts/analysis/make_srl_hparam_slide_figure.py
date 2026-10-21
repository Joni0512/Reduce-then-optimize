"""
2026-10-13: hyperparameter slide figure (tuned vs. fixed), see chat. Top: table of the tuned parameters (actor_lr, critic_lr, seed) with the finding, next to a scatter
of actor_lr vs. best validation service rate of the 27 finished runs with a result (wandb sweeps v1-v4 + repeats of configs D and E, collapsed runs with 0.0 excluded;
RHO val 0.7043 dashed). Bottom: table of the fixed parameters with the reason ("default" where no reason is documented). Run data are cached in
outputs/hparam_runs.json (first run reads wandb + outputs/srl_training_sweep/best[DE]*_s*/result.json). Writes figures_export/SRL/srl_hparam_slide.{png,pdf}.
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
CACHE = REPO / "outputs" / "hparam_runs.json"
INK, GREY, BLUE, LIGHT = "#2c2c2a", "#6b6b66", "#0072B2", "#e6f1fb"

if CACHE.exists():
    runs = json.load(open(CACHE))
else:
    import wandb
    api = wandb.Api(); runs = []
    for f in ["sweep_srl_buffered_id.txt", "sweep_srl_buffered_v2_id.txt", "sweep_srl_buffered_v3_id.txt", "sweep_srl_buffered_v4_id.txt"]:
        for r in api.sweep((REPO / f).read_text().strip()).runs:
            s, c = r.summary, r.config
            if s.get("best_val_service_rate") is not None and c.get("actor_lr") and c.get("critic_lr"):
                runs.append({"src": "sweep", "actor_lr": c["actor_lr"], "critic_lr": c["critic_lr"], "best_val": s["best_val_service_rate"]})
    for p in glob.glob(str(REPO / "outputs/srl_training_sweep/best[DE]*_s*/result.json")):
        d = json.load(open(p))
        runs.append({"src": "repeat_D" if abs(d["actor_lr"] - 0.0031923) < 1e-6 else "repeat_E", "actor_lr": d["actor_lr"], "critic_lr": d["critic_lr"], "best_val": d["best_val_service_rate"]})
    runs = [r for r in runs if r["best_val"] > 0.05]
    json.dump(runs, open(CACHE, "w"), indent=1)
rho_val = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json"))) / 7

fig = plt.figure(figsize=(16, 9))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 160); bg.set_ylim(0, 90); bg.axis("off")


def t(x, y, s, fs=11, w="normal", c=INK, ha="left", va="center", st_="normal"):
    bg.text(x, y, s, fontsize=fs, weight=w, color=c, ha=ha, va=va, style=st_)


t(3, 86, "SRL training: hyperparameters", fs=18, w="bold")

# --- tuned block
bg.add_patch(plt.Rectangle((3, 71), 54, 7, fc=BLUE, ec="none")); t(5, 74.5, "TUNED", fs=12, w="bold", c="white")
t(22, 74.5, "value / range", fs=10, c="white", ha="left")
cols_x = [5, 22, 5]  # parameter, range; finding on the next line
rows = [
    ("actor_lr", "0.0015 to 0.0128 (log-uniform)", "best at about 0.003 to 0.008; collapses above about 0.025"),
    ("critic_lr", "0.0008 to 0.002 (log-uniform)", "no clear effect in this range"),
    ("seed", "random", "large spread between seeds (std 0.015 to 0.06)"),
]
y = 66.5
for name, rng, finding in rows:
    t(5, y, name, fs=12, w="bold"); t(22, y, rng, fs=10.5, c=GREY)
    t(5, y - 3.4, "→ " + finding, fs=10, w="bold", c=BLUE)
    bg.plot([3, 57], [y - 5.5, y - 5.5], color="#d0d0cc", lw=0.8)
    y -= 7.6

# scatter of actor_lr vs best val
ax = fig.add_axes([0.45, 0.52, 0.52, 0.31])
ax.axvspan(0.003, 0.008, color=LIGHT, zorder=0)
for key, col, lab in [("sweep", "#9a9a94", "sweep runs"), ("repeat_D", "#0072B2", "repeats, actor_lr 0.0032 (config D)"), ("repeat_E", "#E69F00", "repeats, actor_lr 0.0093 (config E)")]:
    sel = [r for r in runs if r["src"] == key]
    ax.scatter([r["actor_lr"] for r in sel], [r["best_val"] for r in sel], s=34, color=col, edgecolor="white", linewidth=0.6, label=f"{lab} (n={len(sel)})", zorder=3)
ax.axhline(rho_val, color="#444444", ls="--", lw=1.4, label=f"RHO validation: {rho_val:.3f}")
from matplotlib.ticker import NullFormatter
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("actor_lr", fontsize=10); ax.set_ylabel("best validation service rate", fontsize=10)
ax.set_xticks([0.002, 0.003, 0.005, 0.008, 0.012]); ax.set_xticklabels(["0.002", "0.003", "0.005", "0.008", "0.012"], fontsize=9)
ax.grid(True, alpha=0.3); ax.legend(fontsize=7.8, loc="lower right", frameon=True)
ax.set_title(f"actor_lr vs. best val ({len(runs)} finished runs, shaded: 0.003 to 0.008)", fontsize=10)

# --- fixed block
bg.add_patch(plt.Rectangle((3, 38), 154, 6, fc="#8a8984", ec="none")); t(5, 41, "FIXED", fs=12, w="bold", c="white")
t(33, 41, "value", fs=10, c="white"); t(52, 41, "role / why", fs=10, c="white")
fixed = [
    ("Softmax temperature", "0.02", "sharpness of the target action; compromise from a Q-spread measurement", True),
    ("Perturbation sigma", "0.2", "noise of the Perturb-and-MAP candidates; default", False),
    ("Samples", "20", "candidates per window; default", False),
    ("Polyak rate", "0.005", "slow target critic (one update per instance); taken over from the earlier TD-bootstrap setup", False),
    ("Discount gamma", "0.99", "horizon of the critic; default", False),
    ("Replay buffer", "40", "ring buffer, bounds how stale old entries get", True),
]
y = 34.2
for name, val, why, cand in fixed:
    t(5, y, name, fs=11.5, w="bold")
    t(33, y, val, fs=11.5, c=INK)
    t(52, y, why, fs=10.5, c=GREY)
    if cand:
        t(154, y, "◇", fs=12, c="#D4A017", ha="center")
    bg.plot([3, 157], [y - 2.6, y - 2.6], color="#d0d0cc", lw=0.8)
    y -= 5.3
t(157, 3.0, "◇ candidate for an ablation", fs=9.5, c="#D4A017", ha="right")
t(3, 3.0, "Rolling horizon batch_interval 200 / step_size 100; 10 critic pretrain + 20 epochs; reward local_positive. Tuned results from sweeps of the buffered variant "
          "(wandb sweeps v1-v4 + repeats of D and E; best val over epochs; collapsed runs excluded).", fs=7.8, c="#444444")
out = OUT / "srl_hparam_slide"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("runs", len(runs), {k: sum(1 for r in runs if r["src"] == k) for k in ("sweep", "repeat_D", "repeat_E")})
