"""
2026-10-09: per-instance validation service rate of the NEW buffered SRL actor (mean over all runs whose per-instance values are
available locally) vs. RHO (batch_interval=200/step_size=100), to see on which Li&Lim validation instances SRL helps (see chat).

Sources:
- SRL: val_curve[].per_instance in outputs/srl_training_sweep/buf*/result.json (8 local runs: 4 single runs with critic_lr
  0.00108/0.001/0.0015/0.002 at actor_lr 0.0032, plus 4 local sweep trials) plus 3 cluster runs (actor_lr 0.0064/0.0128 scan runs and
  the serial sweep trial, values pasted from the cluster); the collapsed actor_lr 0.0255 run is excluded (all zeros).
- RHO: mean over 7 repeated RHO runs (keep_active=False) of outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42 (see below).
Two SRL readings per instance: (a) val at epoch 20 (comparable for all runs), (b) val at each run's best-val epoch.
n=1 per run; no run collapsed among the 8. Writes PNG + PDF (+ .tex) into figures_export/SRL/.
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures_export" / "SRL"
VAL = ["lc103", "lc202", "lr103", "lr107", "lr202", "lr208", "lrc102", "lrc202", "lrc205"]

# 2026-10-09: RHO reference = MEAN over the 7 repeated RHO runs (keep_active=False, cardinality 2, seed 42, batch_interval=200/
# step_size=100) in outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run{1..7}/summary.json (runs 1-3 existed, runs 4-7
# were added today because RHO itself fluctuates between runs); keep_active=False = same commitment rule as the SRL training loop.
# The earlier single-run reference (rho_val_bi200_rerun.log, mean 0.6891, keep_active unknown) is kept only as "rho_old" for the print-out.
rho_runs = [json.load(open(p))["rates"] for p in sorted(glob.glob(str(REPO / "outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json")))]
rho = {i: st.mean(r[i] for r in rho_runs) for i in VAL}
rho_std = {i: st.pstdev([r[i] for r in rho_runs]) for i in VAL}
rho_old = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(\w+): rho_service_rate=([0-9.]+)", (REPO / "rho_val_bi200_rerun.log").read_text(), re.M)}
print(f"RHO: {len(rho_runs)} runs, per-run means:", [round(st.mean(r.values()), 4) for r in rho_runs])

runs = []
for p in sorted(glob.glob(str(REPO / "outputs/srl_training_sweep/buf*/result.json"))):
    d = json.load(open(p))
    curve = {pt["epoch"]: pt["per_instance"] for pt in d["val_curve"]}
    runs.append({"name": Path(p).parent.name, "e20": curve[20], "best": curve[d["best_epoch"]], "best_epoch": d["best_epoch"]})


# 2026-10-09: cluster runs whose result.json only exist on the cluster - per-instance val values copied from the cluster print-out the user
# pasted into the chat on 2026-10-09 (e20 = val at epoch 20, best = val at the run's best-val epoch). The collapsed actor_lr=0.0255 run
# (buf_ep20_ps_alr0p0255_s42: all instances 0.0 at every validation) is deliberately NOT averaged in; it is reported separately.
CLUSTER_RUNS = [
    {"name": "buf_ep20_ps_alr0p0064_s42", "best_epoch": 20,
     "e20": {"lc103": 0.7885, "lc202": 0.7255, "lr103": 0.8077, "lr107": 0.6538, "lr202": 0.7, "lr208": 0.54, "lrc102": 0.7547, "lrc202": 0.6471, "lrc205": 0.7255}},
    {"name": "buf_ep20_ps_alr0p0128_s42", "best_epoch": 15,
     "e20": {"lc103": 0.8269, "lc202": 0.7451, "lr103": 0.6731, "lr107": 0.5769, "lr202": 0.62, "lr208": 0.5, "lrc102": 0.6604, "lrc202": 0.549, "lrc205": 0.6275},
     "best": {"lc103": 0.7885, "lc202": 0.7451, "lr103": 0.7885, "lr107": 0.6154, "lr202": 0.58, "lr208": 0.6, "lrc102": 0.717, "lrc202": 0.6078, "lrc205": 0.7255}},
    {"name": "bufsweep_serial_b29579a3", "best_epoch": 15,
     "e20": {"lc103": 0.9038, "lc202": 0.8235, "lr103": 0.7885, "lr107": 0.6538, "lr202": 0.62, "lr208": 0.62, "lrc102": 0.717, "lrc202": 0.6078, "lrc205": 0.7451},
     "best": {"lc103": 0.9038, "lc202": 0.902, "lr103": 0.7885, "lr107": 0.6923, "lr202": 0.56, "lr208": 0.6, "lrc102": 0.7736, "lrc202": 0.6471, "lrc205": 0.6863}},
]
for c in CLUSTER_RUNS:
    c.setdefault("best", c["e20"])  # best epoch == 20 for the alr 0.0064 run
    runs.append(c)


def agg(key):
    out = {}
    for inst in VAL:
        vals = [r[key][inst] for r in runs if inst in r[key]]
        out[inst] = (st.mean(vals), st.pstdev(vals), min(vals), max(vals), sum(v > rho[inst] for v in vals), len(vals))
    return out


A20, ABEST = agg("e20"), agg("best")
print(f"{len(runs)} SRL runs")
print("inst   RHO   | ep20: mean  std  min  max  #>RHO | best-ep: mean  #>RHO")
for i in VAL:
    a, b = A20[i], ABEST[i]
    print(f"{i:6s} {rho[i]:.3f} | {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} {a[3]:.3f} {a[4]}/{a[5]} | {b[0]:.3f} {b[4]}/{b[5]}  diff20 {a[0]-rho[i]:+.3f}")
print("means: RHO(7 runs) %.4f | SRL ep20 %.4f | SRL best-ep %.4f" % (st.mean(rho.values()), st.mean(A20[i][0] for i in VAL), st.mean(ABEST[i][0] for i in VAL)))

# bar chart: RHO vs SRL mean (epoch 20) +- std, dots = single runs
import numpy as np
x = np.arange(len(VAL))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - 0.2, [rho[i] for i in VAL], 0.38, yerr=[rho_std[i] for i in VAL], capsize=3, color="#444444", label=f"RHO, mean of {len(rho_runs)} runs (±1 std), batch_interval=200/step_size=100, keep_active=False")
ax.bar(x + 0.2, [A20[i][0] for i in VAL], 0.38, yerr=[A20[i][1] for i in VAL], capsize=3, color="#0072B2", label=f"SRL buffered, mean of {len(runs)} runs at epoch 20 (±1 std)")
for k, i in enumerate(VAL):
    ax.scatter([k + 0.2] * len([r for r in runs]), [r["e20"][i] for r in runs], s=10, color="white", edgecolor="#003b5c", zorder=3, linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(VAL)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Validation service rate")
ax.set_title("Per validation instance: SRL (buffered actor update) vs. RHO", fontsize=12)
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, axis="y", alpha=0.3)
fig.text(0.01, 0.005, "Li&Lim validation instances; SRL: twin critic, no warmstart, pickup_slack, batch_interval=200/step_size=100, 10 pretrain + 20 epochs, dots = single runs. "
         "Sources: result.json val_curve.per_instance (local runs), outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run1-7 (RHO).", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_vs_rho_per_instance_val"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)

rows = [[i, f"{rho[i]:.3f}", f"{A20[i][0]:.3f}", f"{A20[i][1]:.3f}", f"{A20[i][2]:.3f}-{A20[i][3]:.3f}", f"{A20[i][0]-rho[i]:+.3f}", f"{A20[i][4]}/{A20[i][5]}", f"{ABEST[i][0]:.3f}", f"{ABEST[i][0]-rho[i]:+.3f}"] for i in VAL]
cols = ["Instance", "RHO", "SRL mean @20", "std", "min-max", "SRL - RHO", "runs > RHO mean", "SRL mean @best", "SRL - RHO"]
tex = ["\\begin{tabular}{lrrrrrrrr}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\hline", "\\end{tabular}"]
(OUT / "srl_vs_rho_per_instance_val.tex").write_text("\n".join(tex) + "\n")
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.auto_set_column_width(col=list(range(len(cols)))); tb.scale(1, 1.4)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0: cell.set_facecolor("#e8e8e8"); cell.set_text_props(weight="bold")
ax.set_title(f"Per-instance validation service rate: SRL (mean of {len(runs)} runs) vs. RHO (mean of {len(rho_runs)} runs), batch_interval=200/step_size=100", fontsize=9)
out = OUT / "srl_vs_rho_per_instance_val_table"
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
