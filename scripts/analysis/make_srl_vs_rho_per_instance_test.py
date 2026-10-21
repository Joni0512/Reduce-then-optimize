"""
2026-10-09: per-instance TEST service rate: buffered SRL (mean over the 8 local runs + 2 cluster scan runs, recomputed from the saved best-val checkpoints by
scripts/analysis/compute_srl_test_per_instance.py -> outputs/srl_test_per_instance/*.json) vs RHO (mean over 5 repeated runs,
keep_active=False, batch_interval=200/step_size=100, outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run1-5).
Plus the two cluster scan runs actor_lr 0.0064/0.0128 (recomputed on the cluster, values from the Slurl log); the serial sweep trial and the collapsed actor_lr 0.0255 run are not included.
n=1 per SRL run; the SRL checkpoint is the best-VAL epoch of each run (test is never used for selection). PNG + PDF + tex in figures_export/SRL/.
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
TEST = ["lc108", "lc205", "lc208", "lr111", "lr205", "lr211", "lrc105", "lrc108", "lrc208"]

rho_runs = [json.load(open(p))["rates"] for p in sorted(glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42_test/run*/summary.json")))]
rho = {i: st.mean(r[i] for r in rho_runs) for i in TEST}
rho_std = {i: st.pstdev([r[i] for r in rho_runs]) for i in TEST}
srl_runs = [json.load(open(p)) for p in sorted(glob.glob(str(REPO / "outputs/srl_test_per_instance/*.json")))]
# 2026-10-09: cluster runs, per-instance test rates recomputed on serial_std by submit_srl_test_per_instance_serial.sbatch (job 5650916; the serial sweep trial was added from the same log) and copied
# from its Slurm log (PERINST lines pasted into the chat). The collapsed actor_lr=0.0255 run (all 0.0) is deliberately NOT averaged in.
CLUSTER_TEST = {
    "buf_ep20_ps_alr0p0064_s42": {"lc108": 0.8113, "lc205": 0.7647, "lc208": 0.7647, "lr111": 0.6481, "lr205": 0.6471, "lr211": 0.44, "lrc105": 0.7778, "lrc108": 0.7115, "lrc208": 0.5686},
    "bufsweep_serial_b29579a3": {"lc108": 0.9623, "lc205": 0.9608, "lc208": 0.9608, "lr111": 0.7037, "lr205": 0.6471, "lr211": 0.56, "lrc105": 0.7778, "lrc108": 0.7115, "lrc208": 0.5882},
    "buf_ep20_ps_alr0p0128_s42": {"lc108": 0.8679, "lc205": 0.7843, "lc208": 0.8039, "lr111": 0.6111, "lr205": 0.6275, "lr211": 0.52, "lrc105": 0.6852, "lrc108": 0.5769, "lrc208": 0.5686},
}
for name, rates in CLUSTER_TEST.items():
    srl_runs.append({"run": name, "per_instance": rates, "n_instances": len(rates)})
assert all(r["n_instances"] == 9 for r in srl_runs), "a run is missing test instances"
srl = {i: [r["per_instance"][i] for r in srl_runs] for i in TEST}

rows = []
print(f"RHO: {len(rho_runs)} runs, mean {st.mean(rho.values()):.4f} | SRL: {len(srl_runs)} runs, mean {st.mean(st.mean(v) for v in srl.values()):.4f}")
for i in TEST:
    m, sd, lo, hi = st.mean(srl[i]), st.pstdev(srl[i]), min(srl[i]), max(srl[i])
    n_above = sum(v > rho[i] + 1e-9 for v in srl[i])
    rows.append([i, f"{rho[i]:.3f}", f"{m:.3f}", f"{sd:.3f}", f"{lo:.2f}-{hi:.2f}", f"{m - rho[i]:+.3f}", f"{n_above}/{len(srl_runs)}"])
    print(f"{i:7s} RHO {rho[i]:.3f} | SRL {m:.3f} std {sd:.3f} [{lo:.3f},{hi:.3f}] diff {m - rho[i]:+.3f} above RHO {n_above}/{len(srl_runs)}")
m_all = st.mean(st.mean(v) for v in srl.values())
rows.append(["Mean", f"{st.mean(rho.values()):.3f}", f"{m_all:.3f}", "", "", f"{m_all - st.mean(rho.values()):+.3f}", ""])

x = np.arange(len(TEST))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - 0.2, [rho[i] for i in TEST], 0.38, yerr=[rho_std[i] for i in TEST], capsize=3, color="#444444", label=f"RHO, mean of {len(rho_runs)} runs (±1 std), batch_interval=200/step_size=100, keep_active=False")
ax.bar(x + 0.2, [st.mean(srl[i]) for i in TEST], 0.38, yerr=[st.pstdev(srl[i]) for i in TEST], capsize=3, color="#0072B2", label=f"SRL buffered, mean of {len(srl_runs)} runs at best-val checkpoint (±1 std)")
for k, i in enumerate(TEST):
    ax.scatter([k + 0.2] * len(srl[i]), srl[i], s=10, color="white", edgecolor="#003b5c", zorder=3, linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(TEST); ax.set_ylim(0, 1.05)
ax.set_ylabel("Test service rate"); ax.grid(True, axis="y", alpha=0.3)
ax.set_title("Per test instance: SRL (buffered actor update) vs. RHO", fontsize=12)
ax.legend(fontsize=8, loc="lower left")
fig.text(0.01, 0.005, "Li&Lim test instances; SRL: twin critic, no warmstart, pickup_slack, batch_interval=200/step_size=100, KEEP_ACTIVE=False, 10 pretrain + 20/30 epochs; dots = single runs. "
         "Sources: outputs/srl_test_per_instance/*.json (recomputed), outputs/rho_val_runs/..._test/run1-5.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_vs_rho_per_instance_test"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

cols = ["Instance", "RHO", "SRL mean", "std", "min-max", "SRL - RHO", "runs > RHO"]
(OUT / "srl_vs_rho_per_instance_test_table.tex").write_text("\n".join(["\\begin{tabular}{lrrrrrr}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\hline", "\\end{tabular}"]) + "\n")
fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9); tb.auto_set_column_width(col=list(range(len(cols)))); tb.scale(1, 1.4)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0: cell.set_facecolor("#e8e8e8"); cell.set_text_props(weight="bold")
ax.set_title(f"Per-instance test service rate: SRL (mean of {len(srl_runs)} runs) vs. RHO (mean of {len(rho_runs)} runs), batch_interval=200/step_size=100", fontsize=9)
out = OUT / "srl_vs_rho_per_instance_test_table"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
