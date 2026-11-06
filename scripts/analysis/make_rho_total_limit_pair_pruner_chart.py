"""
2026-10-22: does the damage of the Pair Pruner depend on the total limit L of requests per vehicle? RHO baseline (no pruner) vs. RHO + Pair Pruner
(GNNv1 rgnn_mixed_c2_pw2_v1_final, threshold 0.5) over L, vehicle capacity fixed to C=2 on board, Li&Lim, cardinality 2, batch_interval=40, step_size=10,
one run per cell, 14 instances (7 class 1, 7 class 2), L in {2,4,6,8,10,12,16,20,none}.

Sources: baseline = outputs/rho_total_limit_sweep_C2_c2_bi40_ss10.csv (lr109, lr208), ..._class1.csv, ..._class2.csv, ..._L16_20_class1.csv, ..._L16_20_class2.csv;
Pair Pruner = outputs/rho_total_limit_sweep_C2_c2_bi40_ss10_pair0.5_class1.csv and ..._class2.csv (incl. pairs_kept / pairs_total from main.log: allowed / candidate request pairs
summed over all windows). RHO is not bit-deterministic: +-1 request (about 2 pp) per cell is noise; differences are paired per instance and L. Runtime is NOT compared
(parallel runs on one laptop distort it).

Run: ./venv/bin/python3 scripts/analysis/make_rho_total_limit_pair_pruner_chart.py
Output (figures_export/LiLim Dataset/): rho_total_limit_pair_pruner_C2_c2_bi40_ss10.{png,pdf}, ..._per_instance.{png,pdf}, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["2", "4", "6", "8", "10", "12", "16", "20", "none"]
S = "outputs/rho_total_limit_sweep_C2_c2_bi40_ss10"


def load(files):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["L"] = d["L"].astype(str)
    d = d[d.status == "ok"].drop_duplicates(["instance", "L"], keep="last")
    d["rate"] = 100 * d["served"] / d["requests"]
    d["cls"] = d["instance"].str.extract(r"^[a-z]+(\d)")[0].map({"1": "Class 1", "2": "Class 2"})
    return d


base = load([S + s for s in (".csv", "_class1.csv", "_class2.csv", "_L16_20_class1.csv", "_L16_20_class2.csv")])
pair = load([S + "_pair0.5_class1.csv", S + "_pair0.5_class2.csv"])
m = base.merge(pair, on=["instance", "L"], suffixes=("_b", "_p"))
m["diff"] = m["rate_p"] - m["rate_b"]
CB, CP = "#1E2761", "#E8A33D"


def main():
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), sharey=True)
    for a, cls in zip(ax, ("Class 1", "Class 2", "All instances")):
        s = m if cls == "All instances" else m[m.cls_b == cls]
        for col, lab, c in (("rate_b", "RHO (no pruner)", CB), ("rate_p", "RHO + Pair Pruner (t=0.5)", CP)):
            y = s.groupby("L")[col].mean().reindex(ORDER)
            a.plot(range(len(ORDER)), y.values, "-o", color=c, linewidth=2.3, markersize=5, label=lab)
        dif = s.groupby("L")["diff"].mean().reindex(ORDER)
        for i, v in enumerate(dif.values):
            a.annotate(f"{v:+.1f}", (i, 3), ha="center", fontsize=8, color="#444444")
        a.set_title(f"{cls}: mean over {s.instance.nunique()} instances (numbers at the bottom = Pair - baseline in pp)", fontsize=9.5)
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(["no limit" if c == "none" else c for c in ORDER])
        a.set_xlabel("Total limit L per vehicle (capacity C=2 on board)", fontsize=10)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(0, 100)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("Service rate (served / requests)", fontsize=10.5)
    ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    kept = pair["pairs_kept"].sum() / pair["pairs_total"].sum() * 100
    fig.suptitle(f"RHO: damage of the Pair Pruner vs. total requests per vehicle L (C=2 on board, cardinality 2, batch_interval=40/step_size=10); Pair Pruner keeps {kept:.0f} % of the candidate pairs", fontsize=11.5)
    fig.text(0.5, 0.012, "Li&Lim, RHO offline, max_cardinality=2, 1 run per cell (+-1 request noise = about 2 pp), equal weight per instance; Pair Pruner GNNv1 rgnn_mixed_c2_pw2_v1_final, threshold 0.5;\n"
             "sources: outputs/rho_total_limit_sweep_C2_c2_bi40_ss10*.csv (baseline), ..._pair0.5_class1.csv / _class2.csv (Pair Pruner)", ha="center", fontsize=7.4, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_total_limit_pair_pruner_C2_c2_bi40_ss10.{ext}", dpi=200)
    plt.close(fig)

    insts = sorted(m.instance.unique(), key=lambda s: (int([c for c in s if c.isdigit()][0]), s))
    fig, axes = plt.subplots(4, 4, figsize=(16, 12.5), sharey=True)
    for a in axes.flat:
        a.set_visible(False)
    for a, inst in zip(axes.flat, insts):
        a.set_visible(True)
        s = m[m.instance == inst].set_index("L").reindex(ORDER)
        a.plot(range(len(ORDER)), s["rate_b"].values, "-o", color=CB, linewidth=1.7, markersize=3.5, label="RHO (no pruner)")
        a.plot(range(len(ORDER)), s["rate_p"].values, "-o", color=CP, linewidth=1.7, markersize=3.5, label="RHO + Pair Pruner")
        pk = pair[pair.instance == inst]
        a.set_title(f"{inst} ({int(s.requests_b.iloc[0])} requests; pairs kept {100 * pk.pairs_kept.sum() / pk.pairs_total.sum():.0f} %)", fontsize=9)
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(["2", "4", "6", "8", "10", "12", "16", "20", "no"], fontsize=7.5)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(0, 105)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=False)
    for a in axes[-1, :]:
        if a.get_visible():
            a.set_xlabel("Total limit L per vehicle", fontsize=9)
    fig.suptitle("RHO: service rate per instance, baseline vs. Pair Pruner (t=0.5), vs. total limit L (C=2 on board, cardinality 2, batch_interval=40/step_size=10)", fontsize=11.5)
    fig.text(0.5, 0.008, "Li&Lim, RHO offline, 1 run per cell (+-1 request noise = about 2 pp); sources: outputs/rho_total_limit_sweep_C2_c2_bi40_ss10*.csv", ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_total_limit_pair_pruner_C2_c2_bi40_ss10_per_instance.{ext}", dpi=170)
    plt.close(fig)

    head = " & ".join([f"$L{{=}}{c}$" if c != "none" else "No limit" for c in ORDER])
    lines = []
    for block, s in (("All 14 instances", m), ("Class 1 (7 instances)", m[m.cls_b == "Class 1"]), ("Class 2 (7 instances)", m[m.cls_b == "Class 2"])):
        lines.append(f"\\multicolumn{{{len(ORDER) + 1}}}{{l}}{{\\textit{{{block}}}}} \\\\")
        for lab, col in (("RHO", "rate_b"), ("RHO + Pair Pruner", "rate_p"), ("Difference (pp)", "diff")):
            y = s.groupby("L")[col].mean().reindex(ORDER)
            lines.append(f"{lab} & " + " & ".join((f"{v:+.1f}" if col == "diff" else f"{v:.1f}") for v in y.values) + r" \\")
        if block != "Class 2 (7 instances)":
            lines.append(r"\midrule")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
           "\\caption{RHO with and without the Pair Pruner (GNNv1, threshold 0.5), cardinality 2, vehicle capacity fixed to $C{=}2$, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10: mean service rate (\\%) "
           "over instances as a function of the total limit $L$ of requests per vehicle. One run per cell, $\\pm$1 request noise, equal weight per instance.}\n"
           "\\begin{tabular}{l" + "c" * len(ORDER) + "}\n\\toprule\n & " + head + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / "rho_total_limit_pair_pruner_C2_c2_bi40_ss10_table.tex").write_text(tex)

    print("cells compared:", len(m), "| pair pruner keeps %.1f %% of candidate pairs overall" % kept)
    for block, s in (("ALL", m), ("Class 1", m[m.cls_b == "Class 1"]), ("Class 2", m[m.cls_b == "Class 2"])):
        t = s.groupby("L")["diff"].agg(["mean", "std", "count"]).reindex(ORDER)
        t["t"] = t["mean"] / (t["std"] / np.sqrt(t["count"]))
        print(f"\n{block}: mean difference Pair - baseline (pp), t over instances")
        print(t.round(2).to_string())
        print("  pooled over all L:", round(s["diff"].mean(), 2), "pp; instance-L cells where pair is lower / equal / higher:",
              int((s["diff"] < 0).sum()), int((s["diff"] == 0).sum()), int((s["diff"] > 0).sum()))
    print("\nkept pairs % per class:", {c: round(100 * pair[pair.cls == c].pairs_kept.sum() / pair[pair.cls == c].pairs_total.sum(), 1) for c in ("Class 1", "Class 2")})


if __name__ == "__main__":
    main()
