"""
2026-10-31: runtime savings of the Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final).
Left: SIL, c3 mixedclass Lite, batch_interval=400/step_size=80, seed 42, epochs 5, learning_rate 0.0001, lite split (34 train / 12 val files):
  baseline (no pruner, job 5670939, sacct Elapsed 4-17:23:51 = 113.40 h, best val 80.26 %) vs. Pair-only (threshold 0.5, job 5670940, sacct Elapsed 1-02:50:31 = 26.84 h,
  best val 78.49 %). Elapsed includes the post-training check over the training files. One seed per variant.
Right: RHO (offline, no learning), C=2 on board, cardinality 2, batch_interval=40/step_size=10, 14 Li&Lim instances, 2 repetitions, base and pair runs interleaved and run
  one at a time (scripts/analysis/rho_pair_pruner_runtime.py): wall-clock of the solver subprocess and sum of the solver's own trip-generation time ("RTV ... generated"),
  Pair Pruner threshold 0.5. Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv and ..._rest.csv.

Run: ./venv/bin/python3 scripts/analysis/make_pair_pruner_runtime_savings_chart.py
Output: figures_export/Pair Pruner SIL/pair_pruner_runtime_savings.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("figures_export/Pair Pruner SIL")
OUT.mkdir(parents=True, exist_ok=True)
CB, CP = "#1E2761", "#E8A33D"

# SIL (sacct Elapsed, hours; best val %)
sil = {"No pruner": (113.40, 80.26), "Pair Pruner (t=0.5)": (26.84, 78.49)}

d = pd.concat([pd.read_csv("outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv"), pd.read_csv("outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10_rest.csv")])
d = d[d.status == "ok"].drop_duplicates(["instance", "variant", "rep"], keep="last")
p = d.pivot_table(index=["instance", "rep"], columns="variant", values=["wall_s", "rtv_s", "served"], aggfunc="first").dropna()
n_pairs = len(p)
rho = {"Total wall-clock": (p[("wall_s", "base")].sum(), p[("wall_s", "pair")].sum()),
       "Trip generation": (p[("rtv_s", "base")].sum(), p[("rtv_s", "pair")].sum())}
kept = d[d.variant == "pair"]
kept_pct = 100 * kept.pairs_kept.sum() / kept.pairs_total.sum()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
labs = list(sil)
vals = [sil[k][0] for k in labs]
a1.bar(range(2), vals, color=[CB, CP], width=0.55, zorder=3)
for i, k in enumerate(labs):
    a1.text(i, vals[i] + 2, f"{vals[i]:.1f} h\nbest val {sil[k][1]:.2f} %", ha="center", fontsize=10, fontweight="bold")
a1.text(0.72, 70, f"{vals[0] / vals[1]:.1f}x faster\n({sil[labs[1]][1] - sil[labs[0]][1]:+.2f} pp best val)", ha="center", fontsize=12, color="#444444")
a1.set_xticks(range(2))
a1.set_xticklabels(labs, fontsize=10)
a1.set_ylabel("Training time (hours, sacct Elapsed)")
a1.set_ylim(0, 135)
a1.set_title("SIL training, c3 mixedclass Lite, batch_interval=400/step_size=80\n(5 epochs, seed 42, 1 run per variant)", fontsize=10)

k = list(rho)
x = range(len(k))
w = 0.36
a2.bar([i - w / 2 for i in x], [100.0] * len(k), width=w, color=CB, zorder=3, label="No pruner (= 100 %)")
rel = [100 * rho[c][1] / rho[c][0] for c in k]
a2.bar([i + w / 2 for i in x], rel, width=w, color=CP, zorder=3, label="Pair Pruner (t=0.5)")
for i, c in enumerate(k):
    a2.text(i - w / 2, 102, f"{rho[c][0]:.0f} s", ha="center", fontsize=9)
    a2.text(i + w / 2, rel[i] + 2, f"{rho[c][1]:.0f} s\n{rho[c][0] / rho[c][1]:.2f}x", ha="center", fontsize=9, fontweight="bold")
a2.set_xticks(list(x))
a2.set_xticklabels(k, fontsize=10)
a2.set_ylabel("Runtime relative to no pruner (%)")
a2.set_ylim(0, 135)
a2.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
a2.set_title(f"RHO, batch_interval=40/step_size=10, C=2, cardinality 2\n({n_pairs} paired runs, 14 instances x 2 reps; pruner keeps {kept_pct:.0f} % of candidate pairs)", fontsize=10)
for a in (a1, a2):
    a.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("Runtime savings of the Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final, threshold 0.5)", fontsize=12.5)
fig.text(0.5, 0.012, "Sources: SIL = sacct Elapsed of jobs 5670939 (baseline) and 5670940 (Pair-only), incl. post-training check; RHO = outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10*.csv, "
         "runs one at a time\n(rho_pair_pruner_runtime.py). Trip generation = solver's own 'RTV ... generated' time (incl. Pair Pruner inference, excl. ILP and simulation).",
         ha="center", fontsize=7.8, color="#666666")
fig.tight_layout(rect=(0, 0.06, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"pair_pruner_runtime_savings.{ext}", dpi=170)
print({c: (round(rho[c][0], 1), round(rho[c][1], 1), round(rho[c][0] / rho[c][1], 2)) for c in rho}, "pairs kept", round(kept_pct, 1), "pairs of runs", n_pairs)
