"""
2026-10-31: RHO only - runtime of RHO without pruner vs. RHO + Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final, threshold 0.5), per instance and in total.
Setup (scripts/analysis/rho_pair_pruner_runtime.py): RHO offline, no learning, C=2 on board (unit-demand manifest copy), no total limit, max_cardinality=2, batch_interval=40,
step_size=10, 14 Li&Lim instances (7 class 1, 7 class 2), 2 repetitions, base and pair run interleaved and one at a time (no parallel jobs).
Runtime = wall-clock seconds of the solver subprocess (time.monotonic); trip generation = sum over windows of the solver's own "feasible trip costs generated" time.
Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv and ..._rest.csv (rep 0 from the first file / rest of the series resumed in the second).

Run: ./venv/bin/python3 scripts/analysis/make_pair_pruner_runtime_rho_chart.py
Output: figures_export/Pair Pruner RHO/pair_pruner_runtime_rho_bi40_ss10.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures_export/Pair Pruner RHO")
CB, CP = "#1E2761", "#E8A33D"

d = pd.concat([pd.read_csv("outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv"), pd.read_csv("outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10_rest.csv")])
d = d[d.status == "ok"].drop_duplicates(["instance", "variant", "rep"], keep="last")
p = d.pivot_table(index=["instance", "rep"], columns="variant", values=["wall_s", "rtv_s", "served", "pairs_kept", "pairs_total"], aggfunc="first")
p = p[p[("wall_s", "base")].notna() & p[("wall_s", "pair")].notna()]
per = pd.DataFrame({"base": p[("wall_s", "base")], "pair": p[("wall_s", "pair")], "rtv_base": p[("rtv_s", "base")], "rtv_pair": p[("rtv_s", "pair")],
                    "kept": p[("pairs_kept", "pair")], "total": p[("pairs_total", "pair")]}).groupby("instance").agg({"base": "mean", "pair": "mean", "rtv_base": "mean", "rtv_pair": "mean",
                                                                                                                      "kept": "sum", "total": "sum"})
per["speedup"] = per.base / per.pair
per["cls"] = [s[len(s.rstrip("0123456789")):][0] for s in per.index]  # first digit of the instance number = class
per = per.sort_values(["cls", "base"])
n_runs = len(p)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [2.4, 1]})
x = np.arange(len(per))
w = 0.38
a1.bar(x - w / 2, per.base, width=w, color=CB, zorder=3, label="RHO (no pruner)")
a1.bar(x + w / 2, per.pair, width=w, color=CP, zorder=3, label="RHO + Pair Pruner (t=0.5)")
for i, (inst, r) in enumerate(per.iterrows()):
    a1.text(i, max(r.base, r.pair) + 0.5, f"{r.speedup:.2f}x", ha="center", fontsize=8.5, fontweight="bold")  # 2026-10-31: label centred above the taller bar of the group
a1.set_xticks(x)
a1.set_xticklabels([f"{i}\n{100 * r.kept / r.total:.0f} %" for i, r in per.iterrows()], fontsize=8.5)
a1.set_xlabel("Instance (second line: share of candidate pairs the Pair Pruner keeps)", fontsize=9)
a1.axvline(per.cls.eq("1").sum() - 0.5, color="#999999", ls=":", lw=1)
a1.text(per.cls.eq("1").sum() / 2 - 0.5, 26, "Class 1", ha="center", fontsize=10, color="#555555")
a1.text(per.cls.eq("1").sum() + per.cls.eq("2").sum() / 2 - 0.5, 26, "Class 2", ha="center", fontsize=10, color="#555555")
a1.set_ylabel("Wall-clock per run (s, mean of 2 repetitions)")
a1.set_title("Per instance (number above each pair of bars = speed-up: time without / with Pair Pruner)", fontsize=10)
a1.set_ylim(0, 27)
a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=9, frameon=False)

tot = {"Total wall-clock": (per.base.sum(), per.pair.sum()), "Trip generation": (per.rtv_base.sum(), per.rtv_pair.sum())}
cls_tot = {f"Class {c}": (per[per.cls == c].base.sum(), per[per.cls == c].pair.sum()) for c in ("1", "2")}
bars = {**tot, **cls_tot}
ks = list(bars)
rel = [100 * bars[k][1] / bars[k][0] for k in ks]
a2.bar(range(len(ks)), [100] * len(ks), width=0.38, color=CB, zorder=3, align="edge", label="no pruner = 100 %")
a2.bar(np.arange(len(ks)) + 0.38, rel, width=0.38, color=CP, zorder=3, align="edge")
for i, k in enumerate(ks):
    a2.text(i + 0.38, 102.5, f"{bars[k][0] / bars[k][1]:.2f}x", ha="center", fontsize=9.5, fontweight="bold")  # 2026-10-31: speed-up above the 100 % bar
a2.set_xticks(np.arange(len(ks)) + 0.38)
a2.set_xticklabels(["Total\nwall-clock", "Trip\ngeneration", "Class 1\nwall-clock", "Class 2\nwall-clock"], fontsize=8.5)
a2.set_ylim(0, 118)
a2.set_ylabel("Runtime relative to no pruner (%)")
a2.set_title("Total over 14 instances", fontsize=10)
for a in (a1, a2):
    a.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
kept = 100 * per.kept.sum() / per.total.sum()
fig.suptitle("RHO: runtime with and without the Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final, t=0.5)\n"
             f"batch_interval=40/step_size=10, C=2 on board, cardinality 2: overall {tot['Total wall-clock'][0] / tot['Total wall-clock'][1]:.2f}x faster, {kept:.0f} % of candidate pairs kept", fontsize=11)
fig.text(0.5, 0.012, f"Li&Lim, RHO offline without learning, 14 instances x 2 repetitions ({n_runs} paired runs), runs one at a time; trip generation = solver's own 'feasible trip costs generated' time "
         "(incl. Pair Pruner inference, excl. ILP and simulation);\nsources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv and ..._rest.csv via scripts/analysis/rho_pair_pruner_runtime.py",
         ha="center", fontsize=7.6, color="#666666")
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"pair_pruner_runtime_rho_bi40_ss10.{ext}", dpi=170)
print(per.round(2).to_string())
print({k: (round(v[0], 1), round(v[1], 1), round(v[0] / v[1], 2)) for k, v in bars.items()})
