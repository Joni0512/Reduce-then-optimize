"""
2026-10-26: controlled runtime comparison RHO (no pruner) vs. RHO + Pair Pruner (GNNv1 rgnn_mixed_c2_pw2_v1_final, t=0.5), measured one run at a time on the laptop
(scripts/analysis/rho_pair_pruner_runtime.py), C=2 on board, no total limit, cardinality 2, batch_interval=40, step_size=10, 14 instances, 2 repetitions per cell.
Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv (44 runs; the first process was stopped when the app quit) and ..._rest.csv (12 runs resumed on 2026-10-26;
rows of the resumed file replace the same (instance, variant, rep) of the first file). wall_s = subprocess wall clock (monotonic), rtv_s = sum over windows of the solver's own
"RTV combos ... generated" time (TripHandler.run incl. Pair Pruner inference, excl. ILP/simulation/start-up).
Run: ./venv/bin/python3 scripts/analysis/make_rho_pair_pruner_runtime_chart.py
Output: figures_export/LiLim Dataset/rho_pair_pruner_runtime_C2_c2_bi40_ss10.{png,pdf} and _table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
P = "outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10"
a, b = pd.read_csv(P + ".csv"), pd.read_csv(P + "_rest.csv")
d = pd.concat([a, b], ignore_index=True).drop_duplicates(["instance", "variant", "rep"], keep="last")
d = d[d.status == "ok"]
g = d.groupby(["instance", "variant"]).agg(wall=("wall_s", "mean"), rtv=("rtv_s", "mean"), served=("served", "mean"), n=("rep", "size")).reset_index()
w = g.pivot(index="instance", columns="variant", values=["wall", "rtv", "served", "n"])
t = pd.DataFrame({"wall_base": w["wall"]["base"], "wall_pair": w["wall"]["pair"], "rtv_base": w["rtv"]["base"], "rtv_pair": w["rtv"]["pair"],
                  "served_base": w["served"]["base"], "served_pair": w["served"]["pair"], "reps": w["n"].min(axis=1)})
t["wall_speedup"] = t.wall_base / t.wall_pair
t["rtv_speedup"] = t.rtv_base / t.rtv_pair
t["cls"] = ["Class 2" if "2" in i[-3:-1] or i[-3] == "2" else "Class 1" for i in t.index]
t["cls"] = [("Class 2" if [c for c in i if c.isdigit()][0] == "2" else "Class 1") for i in t.index]
t = t.loc[sorted(t.index, key=lambda s: (int([c for c in s if c.isdigit()][0]), s))]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [3, 1.3]})
    x = np.arange(len(t))
    ax[0].bar(x - 0.2, t.wall_base, 0.4, color="#1E2761", label="RHO (no pruner)")
    ax[0].bar(x + 0.2, t.wall_pair, 0.4, color="#E8A33D", label="RHO + Pair Pruner (t=0.5)")
    for i, (wb, wp, sp) in enumerate(zip(t.wall_base, t.wall_pair, t.wall_speedup)):
        ax[0].annotate(f"{sp:.2f}x", (i, max(wb, wp)), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(t.index, rotation=45, ha="right")
    ax[0].set_ylabel("Wall clock per run (s, mean of 2 repetitions)")
    ax[0].set_yscale("log")
    ax[0].set_title("Per instance (number = speedup base / pair)", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=8.5)
    ax[0].grid(axis="y", alpha=0.3)
    cls = t.groupby("cls")[["wall_base", "wall_pair", "rtv_base", "rtv_pair"]].sum()
    cls.loc["All"] = cls.sum()
    xx = np.arange(len(cls))
    ax[1].bar(xx - 0.2, cls.wall_base / 60, 0.4, color="#1E2761")
    ax[1].bar(xx + 0.2, cls.wall_pair / 60, 0.4, color="#E8A33D")
    for i, (wb, wp) in enumerate(zip(cls.wall_base, cls.wall_pair)):
        ax[1].annotate(f"{wb / wp:.2f}x", (i, max(wb, wp) / 60), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(cls.index)
    ax[1].set_ylabel("Sum of mean wall clock (min)")
    ax[1].set_title("Total per group (number = speedup)", fontsize=10.5)
    ax[1].grid(axis="y", alpha=0.3)
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle("RHO: runtime with and without the Pair Pruner, measured one run at a time (C=2 on board, no total limit, cardinality 2, batch_interval=40/step_size=10)", fontsize=11.5)
    fig.text(0.5, 0.01, "Li&Lim, 14 instances, 2 repetitions per cell, wall clock = solver subprocess (monotonic clock). Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi40_ss10.csv + ..._rest.csv", ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_pair_pruner_runtime_C2_c2_bi40_ss10.{ext}", dpi=190)
    plt.close(fig)
    lines = [f"{i.replace('_', chr(92) + '_')} & {r.wall_base:.1f} & {r.wall_pair:.1f} & {r.wall_speedup:.2f} & {r.rtv_base:.1f} & {r.rtv_pair:.1f} & {r.rtv_speedup:.2f} \\\\" for i, r in t.iterrows()]
    for name in ("Class 1", "Class 2"):
        s = t[t.cls == name]
        lines.append(f"\\textbf{{{name} (sum)}} & {s.wall_base.sum():.1f} & {s.wall_pair.sum():.1f} & {s.wall_base.sum() / s.wall_pair.sum():.2f} & {s.rtv_base.sum():.1f} & {s.rtv_pair.sum():.1f} & {s.rtv_base.sum() / s.rtv_pair.sum():.2f} \\\\")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n\\caption{RHO (cardinality 2, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, capacity $C{=}2$, no total limit): runtime without and with the Pair Pruner (GNNv1, threshold 0.5), "
           "measured one run at a time, mean of 2 repetitions. Wall = subprocess wall clock, RTV = solver time for trip generation incl.\\ pruner inference (seconds).}\n"
           "\\begin{tabular}{lrrrrrr}\n\\toprule\nInstance & Wall base & Wall pair & Speedup & RTV base & RTV pair & Speedup \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / "rho_pair_pruner_runtime_C2_c2_bi40_ss10_table.tex").write_text(tex)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string())
    for name, s in (("Class 1", t[t.cls == "Class 1"]), ("Class 2", t[t.cls == "Class 2"]), ("All", t)):
        print(f"{name}: wall base {s.wall_base.sum():.0f}s pair {s.wall_pair.sum():.0f}s speedup {s.wall_base.sum() / s.wall_pair.sum():.2f}x | RTV base {s.rtv_base.sum():.0f}s pair {s.rtv_pair.sum():.0f}s speedup {s.rtv_base.sum() / s.rtv_pair.sum():.2f}x | served base {s.served_base.sum():.0f} pair {s.served_pair.sum():.0f}")
    print("cells with fewer than 2 repetitions:", int((t.reps < 2).sum()))


if __name__ == "__main__":
    main()
