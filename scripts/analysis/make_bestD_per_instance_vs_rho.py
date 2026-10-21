"""
2026-10-13: per-instance VALIDATION service rate of config D (actor_lr 0.0031923396291543868, critic_lr 0.002, 20 epochs, MLP actor, seeds 301-305) vs RHO.
One panel, 3 bars per instance (2026-10-13, user request): RHO, SRL at each seed's best-val epoch, SRL at epoch 20 (result.json val_curve.per_instance). SRL bars = mean +-1 std over the 5 seeds,
dots = single seeds. RHO (keep_active=False, batch_interval=200/step_size=100): mean +-1 std over 7 validation runs, outputs/rho_val_runs.
Writes figures_export/SRL/bestD_per_instance_vs_rho.{png,pdf} + .tex table and prints the numbers. (A test-instance variant was dropped: user wants val only.)
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
seeds = [301, 302, 303, 304, 305]
colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
VAL = ["lc103", "lc202", "lr103", "lr107", "lr202", "lr208", "lrc102", "lrc202", "lrc205"]

rho_runs = [json.load(open(p))["rates"] for p in sorted(glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json")))]
rho = {i: st.mean(r[i] for r in rho_runs) for i in VAL}
rho_sd = {i: st.pstdev([r[i] for r in rho_runs]) for i in VAL}

e20, best, best_ep = {}, {}, {}
for s in seeds:
    d = json.load(open(SW / f"bestD_s{s}" / "result.json"))
    curve = {pt["epoch"]: pt["per_instance"] for pt in d["val_curve"]}
    e20[s], best[s], best_ep[s] = curve[20], curve[d["best_epoch"]], d["best_epoch"]
print("best epochs:", best_ep)

# 2026-10-13: extra group "mean" at the end (user request): mean over the 9 instances; error bars/dots = spread of the per-run means (RHO: 7 runs, SRL: 5 seeds)
LABELS = VAL + ["mean"]
x = np.arange(len(LABELS))
fig, ax = plt.subplots(figsize=(14, 6))
w = 0.26
mean_best = {i: st.mean(best[s][i] for s in seeds) for i in VAL}; sd_best = {i: st.pstdev([best[s][i] for s in seeds]) for i in VAL}
mean_20 = {i: st.mean(e20[s][i] for s in seeds) for i in VAL}; sd_20 = {i: st.pstdev([e20[s][i] for s in seeds]) for i in VAL}
rho_run_means = [st.mean(r.values()) for r in rho_runs]
seed_means_best = [st.mean(best[s][i] for i in VAL) for s in seeds]
seed_means_20 = [st.mean(e20[s][i] for i in VAL) for s in seeds]
rho_y = [rho[i] for i in VAL] + [st.mean(rho_run_means)]; rho_e = [rho_sd[i] for i in VAL] + [st.pstdev(rho_run_means)]
best_y = [mean_best[i] for i in VAL] + [st.mean(seed_means_best)]; best_e = [sd_best[i] for i in VAL] + [st.pstdev(seed_means_best)]
e20_y = [mean_20[i] for i in VAL] + [st.mean(seed_means_20)]; e20_e = [sd_20[i] for i in VAL] + [st.pstdev(seed_means_20)]
ax.bar(x - w, rho_y, w, yerr=rho_e, capsize=3, color="#444444", label=f"RHO, mean of {len(rho_runs)} runs (±1 std): {st.mean(rho.values()):.3f}")
ax.bar(x, best_y, w, yerr=best_e, capsize=3, color="#56B4E9", label=f"SRL, best epoch of each seed, mean of 5 seeds: {st.mean(mean_best.values()):.3f}")
ax.bar(x + w, e20_y, w, yerr=e20_e, capsize=3, color="#0072B2", label=f"SRL, epoch 20, mean of 5 seeds: {st.mean(mean_20.values()):.3f}")
for k, i in enumerate(LABELS):
    for j, (s, c) in enumerate(zip(seeds, colors)):
        vb = best[s][i] if i != "mean" else seed_means_best[j]
        v2 = e20[s][i] if i != "mean" else seed_means_20[j]
        ax.scatter(k + (j - 2) * 0.045, vb, s=13, color=c, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(k + w + (j - 2) * 0.045, v2, s=13, color=c, edgecolor="white", linewidth=0.4, zorder=3)
for k, (a, b_, c_) in enumerate(zip(rho_y, best_y, e20_y)):
    if k == len(VAL):
        for off, v in ((-w, a), (0, b_), (w, c_)):
            ax.text(k + off, 0.205, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="white", weight="bold", rotation=90)
ax.axvline(len(VAL) - 0.5, color="#999999", ls=":", lw=1.2)
for s, c in zip(seeds, colors):
    ax.scatter([], [], s=13, color=c, label=f"seed {s} (dots)")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_xlim(-0.6, len(LABELS) - 0.4); ax.set_ylim(0.2, 1.0); ax.set_ylabel("Validation service rate"); ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
ax.set_title("SRL (actor_lr=0.003192, critic_lr=0.002), 5 seeds: per-instance validation service rate vs. RHO", fontsize=11.5)
print("best epochs:", best_ep)
for i in VAL:
    print(f"{i:6s} RHO {rho[i]:.3f}  best {mean_best[i]:.3f}  ep20 {mean_20[i]:.3f}")
fig.text(0.01, 0.005, "Li&Lim validation instances, batch_interval=200/step_size=100; SRL: buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 pretrain + 20 epochs. "
         "Sources: outputs/srl_training_sweep/bestD_s301-305/result.json (val_curve.per_instance); RHO: outputs/rho_val_runs (keep_active=False, 7 runs).", fontsize=6.3, color="#444444")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = OUT / "bestD_per_instance_vs_rho"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
cols = ["Instance", "RHO", "SRL best epoch", "SRL epoch 20", "SRL (ep 20) - RHO"]
rows = [[i, f"{rho[i]:.3f}", f"{mean_best[i]:.3f}", f"{mean_20[i]:.3f}", f"{mean_20[i]-rho[i]:+.3f}"] for i in VAL] + [["mean", f"{st.mean(rho.values()):.3f}", f"{st.mean(mean_best.values()):.3f}", f"{st.mean(mean_20.values()):.3f}", f"{st.mean(mean_20.values())-st.mean(rho.values()):+.3f}"]]
tex = ["\\begin{tabular}{lrrrr}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\hline", "\\end{tabular}"]
(OUT / "bestD_per_instance_vs_rho.tex").write_text("\n".join(tex) + "\n")
