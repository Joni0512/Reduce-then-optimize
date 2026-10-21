"""
2026-10-13: config D (actor_lr 0.0031923396291543868, critic_lr 0.002, 20 epochs, seeds 301-305, buffered actor update batch 16, twin critic, no warmstart,
pickup_slack, batch_interval=200/step_size=100) vs RHO. Left: validation service rate over the epochs (per seed, mean +-1 std); right: val at epoch 20,
best val over the epochs and test (best-val checkpoint) per seed + mean +-std. RHO references (keep_active=False): validation = mean of 7 runs,
test = mean of 5 runs, from outputs/rho_val_runs. Sources: outputs/srl_training_sweep/bestD_s<seed>/result.json. PNG + PDF in figures_export/SRL/.
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
seeds = [301, 302, 303, 304, 305]
res = {s: json.load(open(SW / f"bestD_s{s}" / "result.json")) for s in seeds}
curves = {s: {r["epoch"]: r["service_rate"] for r in res[s]["val_curve"]} for s in seeds}
epochs = sorted(curves[seeds[0]])
rho_val = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json"))) / 7
rho_test = sum(json.load(open(p))["mean"] for p in glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run*/summary.json"))) / 5
colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
for s, col in zip(seeds, colors):
    ax1.plot(epochs, [curves[s][e] for e in epochs], marker="o", linewidth=1.8, markersize=6, color=col, alpha=0.9, label=f"seed {s}")
mean = [st.mean(curves[s][e] for s in seeds) for e in epochs]
sd = [st.stdev(curves[s][e] for s in seeds) for e in epochs]
ax1.plot(epochs, mean, color="black", linewidth=3.2, marker="s", markersize=7, label="mean over 5 seeds")
ax1.fill_between(epochs, [m - x for m, x in zip(mean, sd)], [m + x for m, x in zip(mean, sd)], color="black", alpha=0.10, label="±1 std")
ax1.axhline(rho_val, color="#444444", linestyle="--", linewidth=1.6, label=f"RHO validation (7 runs): {rho_val:.4f}")
for e, m in zip(epochs, mean):
    ax1.text(e + 0.25, m - 0.012, f"{m:.3f}", fontsize=8)
ax1.set_xticks(epochs); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation service rate (mean over 9 instances)")
ax1.set_ylim(0.20, 0.80); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("Validation over the epochs", fontsize=11)

metrics = [("Val at epoch 20", [curves[s][20] for s in seeds], rho_val, "#0072B2"),
           ("Best val (any epoch)", [res[s]["best_val_service_rate"] for s in seeds], rho_val, "#0072B2"),
           ("Test (best-val checkpoint)", [res[s]["test_service_rate"] for s in seeds], rho_test, "#E69F00")]
for i, (lab, vals, ref, col) in enumerate(metrics):
    m, sdv = st.mean(vals), st.stdev(vals)
    ax2.bar(i, m, width=0.55, color=col, alpha=0.55, yerr=sdv, capsize=6, ecolor="black")
    ax2.plot([i - 0.275, i + 0.275], [ref, ref], color="#444444", linestyle="--", linewidth=2)
    for s, v, c in zip(seeds, vals, colors):
        ax2.scatter(i + (seeds.index(s) - 2) * 0.09, v, color=c, s=40, zorder=3)
    ax2.text(i, 0.628, f"{m:.3f} ± {sdv:.3f}", ha="center", fontsize=10, weight="bold")
    ax2.text(i, ref + 0.003, f"RHO {ref:.4f}", ha="center", fontsize=8, color="#444444")
ax2.set_xticks(range(3)); ax2.set_xticklabels([m[0] for m in metrics], fontsize=9)
ax2.set_ylim(0.62, 0.80); ax2.set_ylabel("Service rate"); ax2.grid(True, axis="y", alpha=0.3)
ax2.set_title("Mean ± std over 5 seeds vs. RHO (dots = seeds, dashed = RHO)", fontsize=11)
fig.suptitle("Config D (actor_lr=0.003192, critic_lr=0.002), 20 epochs, seeds 301-305, vs. RHO", fontsize=13)
fig.text(0.01, 0.005, "Buffered actor update (batch 16), twin critic, no warmstart, pickup_slack, 10 critic pretrain + 20 epochs, validation every 5 epochs, batch_interval=200/step_size=100. "
         "Sources: outputs/srl_training_sweep/bestD_s301-305/result.json; RHO: outputs/rho_val_runs (keep_active=False; val 7 runs, test 5 runs).", fontsize=6.3, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
out = OUT / "bestD_5seeds_vs_rho"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
for lab, vals, ref, _ in metrics:
    print(lab, [round(v, 4) for v in vals], "mean", round(st.mean(vals), 4), "std", round(st.stdev(vals), 4), "RHO", round(ref, 4))
