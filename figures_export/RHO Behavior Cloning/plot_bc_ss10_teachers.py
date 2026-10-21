"""2026-10-10: state-sync BC at step_size=10 - actor batch_interval=20 vs RHO teachers batch_interval=40/60/80
(all step_size=10, card 2, seed 42, random init, 20 epochs, lr 1e-4, val every 5 epochs on 9 VAL_INSTANCES, KEEP_ACTIVE=True).
Sources (LRZ jobs, filtered logs copied to ~/cluster_logs):
  teacher 40: bc_sync_ss10_rho40_5638673_filtered.log
  teacher 60: bc_sync_ss10_rho60_5646338_filtered.log (rerun; 5638674 died after epoch 5 on a Gurobi token-server error)
  teacher 80: bc_sync_ss10_rho80_5638675_filtered.log
RHO val references (same 9 VAL_INSTANCES, step_size=10, card 2, KEEP_ACTIVE=True, 10 runs = 5 repeats + 5 seeds):
  rho_val_bi{20,40,60,80}_ss10_card2*.log in the repo root."""
import re, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
LOGS = {40: "bc_sync_ss10_rho40_5638673_filtered.log", 60: "bc_sync_ss10_rho60_5646338_filtered.log", 80: "bc_sync_ss10_rho80_5638675_filtered.log"}
def rho_val(bi):
    runs = [np.mean([float(m[1]) for m in (re.search(r"\] \w+: ([0-9.]+) \(", l) for l in open(p)) if m])
            for p in sorted(glob.glob(str(root / f"rho_val_bi{bi}_ss10_card2*.log")))]
    return float(np.mean(runs)), float(np.std(runs)), len(runs)
curves = {}
for t, f in LOGS.items():
    txt = open(Path.home() / "cluster_logs" / f).read()
    curves[t] = {int(e): float(v) for e, v in re.findall(r"epoch (\d+): val=([0-9.]+)", txt)}
ref20 = rho_val(20)
print("teacher actor_val_by_epoch best last teacher_val(mean,std,n)")
for t in LOGS:
    print(t, curves[t], max(curves[t].values()), curves[t][20], rho_val(t))
print("RHO batch_interval=20 (actor's own horizon):", ref20)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.2, 1]})
x = np.arange(len(LOGS)); w = 0.26
best = [max(curves[t].values()) for t in LOGS]; last = [curves[t][20] for t in LOGS]; tv = [rho_val(t) for t in LOGS]
for k, (vals, col, lab) in enumerate([(best, "#2a78d6", "actor: best val epoch"), (last, "#9db8d9", "actor: epoch 20"),
                                      ([m for m, s, n in tv], "#8a8984", "RHO teacher (10 runs)")]):
    a1.bar(x + (k - 1) * w, vals, w, color=col, label=lab, zorder=2,
           yerr=[s for m, s, n in tv] if k == 2 else None, capsize=2)
    errs = [s for m, s, n in tv] if k == 2 else [0] * len(vals)  # 2026-10-10: label above the error bar
    for xi, v, e in zip(x + (k - 1) * w, vals, errs):
        a1.text(xi, v + e + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
a1.axhline(ref20[0], color="#eb6834", ls="--", lw=1, zorder=3, label=f"RHO batch_interval=20 = actor's horizon ({ref20[0]:.3f})")
a1.set_xticks(x, [f"teacher batch_interval={t}" for t in LOGS], fontsize=8.5); a1.set_ylim(0.65, 0.92)
a1.set_ylabel("service rate (9 val instances)", fontsize=9); a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
a1.set_title("Actor (batch_interval=20) vs its RHO teacher", loc="left", fontsize=10)
for t, c in zip(LOGS, ["#2a78d6", "#1baf7a", "#eda100"]):
    pass
# 2026-10-10: per-epoch values as grouped bars instead of lines (user request)
EPS = [5, 10, 15, 20]; xe = np.arange(len(EPS)); we = 0.26
for k, (t, c) in enumerate(zip(LOGS, ["#2a78d6", "#1baf7a", "#eda100"])):
    vals = [curves[t][e] for e in EPS]
    a2.bar(xe + (k - 1) * we, vals, we, color=c, label=f"teacher batch_interval={t}", zorder=2)
    for xi, v in zip(xe + (k - 1) * we, vals):
        a2.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
a2.axhline(ref20[0], color="#eb6834", ls="--", lw=1, zorder=3, label="RHO batch_interval=20")
a2.set_xticks(xe, [str(e) for e in EPS]); a2.set_xlabel("epoch", fontsize=9); a2.set_ylim(0.65, 0.92)
a2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
a2.set_title("Actor val service rate per epoch (1 run each)", loc="left", fontsize=10)
for a in (a1, a2):
    a.yaxis.grid(True, color=GRID, zorder=0); a.set_axisbelow(True); a.tick_params(labelsize=8)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("State-sync BC at step_size=10: actor batch_interval=20 learning from RHO batch_interval=40/60/80", x=0.01, y=0.985, ha="left", fontsize=12)
fig.text(0.01, 0.915, "all step_size=10, card 2, seed 42, random init, 20 epochs, lr 1e-4, val every 5 epochs (KEEP_ACTIVE=True); actor: 1 run per teacher; y-axis starts at 0.65",
         fontsize=8, color=INK2)
fig.text(0.01, 0.005, "Sources: ~/cluster_logs/bc_sync_ss10_rho{40_5638673,60_5646338,80_5638675}_filtered.log; RHO: rho_val_bi*_ss10_card2*.log", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(here / f"bc_ss10_actor_vs_teachers.{ext}", dpi=200)
