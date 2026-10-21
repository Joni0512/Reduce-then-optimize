"""
2026-10-09: overall comparison buffered SRL vs RHO (validation and test, pooled over the 9 instances of each split), see chat.
SRL: all finished runs of the new method (single/scan runs and finished sweep trials, via scripts/analysis/make_all_finished_runs_table.py). The collapsed actor_lr=0.0255 run is excluded from the SRL
statistics (reported separately). RHO: mean over the repeated runs with keep_active=False, batch_interval=200/step_size=100:
7 validation runs (outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42) and 5 test runs (..._test).
n=1 per SRL run; the SRL test rate is measured at each run's best-val checkpoint. PNG + PDF + tex in figures_export/SRL/.
"""
import glob
import json
import runpy
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
rows = runpy.run_path(str(REPO / "scripts/analysis/make_all_finished_runs_table.py"))["ROWS"]
srl = [r for r in rows if r["method"].startswith("new")]
collapsed = [r for r in srl if r["best_val"] < 0.1]
srl = [r for r in srl if r["best_val"] >= 0.1]
# 2026-10-09: the finished sweep trials are now part of RUNS in make_buffered_actor_results_overview.py (no extra list needed here)
metrics = {"Val @ epoch 20": [r["val"][20] for r in srl], "Val @ best epoch": [r["best_val"] for r in srl], "Test @ best-val checkpoint": [r["test"] for r in srl]}


def rho_means(pattern):
    return [json.load(open(p))["mean"] for p in sorted(glob.glob(str(REPO / pattern)))]


rho_val = rho_means("outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json")
rho_test = rho_means("outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run*/summary.json")
rho = {"Val @ epoch 20": rho_val, "Val @ best epoch": rho_val, "Test @ best-val checkpoint": rho_test}

stats = []
for k, v in metrics.items():
    stats.append([k, f"{st.mean(rho[k]):.4f} (n={len(rho[k])}, std {st.pstdev(rho[k]):.4f})", f"{st.mean(v):.4f}", f"{st.median(v):.4f}", f"{st.pstdev(v):.4f}", f"{min(v):.3f}-{max(v):.3f}",
                  f"{st.mean(v) - st.mean(rho[k]):+.4f}", f"{sum(x > st.mean(rho[k]) for x in v)}/{len(v)}"])
    print(stats[-1])
n_all = len(srl) + len(collapsed)
test_all = st.mean([r["test"] for r in srl] + [r["test"] for r in collapsed])
print(f"{len(srl)} SRL runs without collapse, {len(collapsed)} collapsed; test mean incl. collapsed {test_all:.4f}")

fig, ax = plt.subplots(figsize=(10, 5.5))
labels = list(metrics)
for k, lab in enumerate(labels):
    ax.bar(k - 0.2, st.mean(rho[lab]), 0.38, yerr=st.pstdev(rho[lab]), capsize=4, color="#444444", label=f"RHO (7 val / 5 test runs, ±1 std)" if k == 0 else None)
    ax.bar(k + 0.2, st.mean(metrics[lab]), 0.38, yerr=st.pstdev(metrics[lab]), capsize=4, color="#0072B2", label=f"SRL buffered, mean of {len(srl)} runs (±1 std)" if k == 0 else None)
    ax.scatter([k + 0.2] * len(metrics[lab]), metrics[lab], s=14, color="white", edgecolor="#003b5c", zorder=3, linewidth=0.7, label="single SRL runs" if k == 0 else None)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylim(0.55, 0.85); ax.set_ylabel("Service rate (mean over 9 instances)"); ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Buffered SRL vs. RHO: validation and test (batch_interval=200/step_size=100)", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
fig.text(0.01, 0.005, f"SRL: twin critic, no warmstart, pickup_slack, KEEP_ACTIVE=False, n=1 per run; {len(collapsed)} collapsed run (actor_lr=0.0255, all 0.0) excluded "
         f"(test mean of all {n_all} runs: {test_all:.3f}). RHO: keep_active=False. Sources: result.json / cluster logs / wandb; outputs/rho_val_runs/.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_vs_rho_overall"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

cols = ["Metric", "RHO mean", "SRL mean", "SRL median", "SRL std", "SRL min-max", "SRL - RHO", "runs > RHO"]
(OUT / "srl_vs_rho_overall_table.tex").write_text("\n".join(["\\begin{tabular}{lrrrrrrr}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"] + [" & ".join(r) + " \\\\" for r in stats] + ["\\hline", "\\end{tabular}"]) + "\n")
fig, ax = plt.subplots(figsize=(13, 2.4)); ax.axis("off")
tb = ax.table(cellText=stats, colLabels=cols, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9); tb.auto_set_column_width(col=list(range(len(cols)))); tb.scale(1, 1.5)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0: cell.set_facecolor("#e8e8e8"); cell.set_text_props(weight="bold")
ax.set_title(f"Buffered SRL ({len(srl)} runs without collapse) vs. RHO, batch_interval=200/step_size=100, keep_active=False", fontsize=9)
out = OUT / "srl_vs_rho_overall_table"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


# 2026-10-10: compact variant with only the two validation views (user request: RHO vs "best" and vs "val 20 epochs")
fig, ax = plt.subplots(figsize=(8, 5.5))
two = ["Val @ best epoch", "Val @ epoch 20"]
for k, lab in enumerate(two):
    ax.bar(k - 0.2, st.mean(rho[lab]), 0.38, yerr=st.pstdev(rho[lab]), capsize=4, color="#444444", label="RHO (7 runs, ±1 std)" if k == 0 else None)
    ax.bar(k + 0.2, st.mean(metrics[lab]), 0.38, yerr=st.pstdev(metrics[lab]), capsize=4, color="#0072B2", label=f"SRL buffered, mean of {len(srl)} runs (±1 std)" if k == 0 else None)
    ax.scatter([k + 0.2] * len(metrics[lab]), metrics[lab], s=14, color="white", edgecolor="#003b5c", zorder=3, linewidth=0.7, label="single SRL runs" if k == 0 else None)
    ax.text(k - 0.2, st.mean(rho[lab]) + 0.012, f"{st.mean(rho[lab]):.3f}", ha="center", fontsize=9)
    ax.text(k + 0.2, st.mean(metrics[lab]) + st.pstdev(metrics[lab]) + 0.012, f"{st.mean(metrics[lab]):.3f}", ha="center", fontsize=9)
ax.set_xticks(range(2)); ax.set_xticklabels(["best validation epoch\n(max over epochs, optimistic)", "validation at epoch 20\n(fixed epoch)"])
ax.set_ylim(0.55, 0.85); ax.set_ylabel("Validation service rate (mean over 9 instances)"); ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Validation: SRL (buffered) vs. RHO", fontsize=12); ax.legend(fontsize=8, loc="upper right")
fig.text(0.01, 0.005, f"SRL: {len(srl)} runs without collapse ({len(collapsed)} collapsed run excluded), twin critic, no warmstart, pickup_slack, batch_interval=200/step_size=100; RHO: keep_active=False.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_vs_rho_best_vs_val20"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
