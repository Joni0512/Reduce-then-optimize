"""
2026-10-22: RHO baseline service rate as a function of the TOTAL limit L of requests per vehicle (--max_requests_per_vehicle), vehicle capacity fixed to C=2
requests on board at the same time, Li&Lim, cardinality 2.

Setup (all runs): RHO offline, no pruner, --max_cardinality 2, --batch_interval 40, --step_size 10, one run per cell; capacity C=2 via manifest copy (am=1, am_capacity=2);
L = Config.MAX_REQUESTS_PER_VEHICLE (completed + on board + newly assigned <= L per vehicle, enforced in CO_TripCostMinimization.solve_ilp); "none" = no limit.
Fleet = number of routes of the reference solution (li_lim_parser), manifests cleared before the run. RHO is not bit-deterministic: +-1 request (about 2 pp) per cell is noise.
Sources (L=16 and 20: ..._L16_20_class1.csv / ..._class2.csv, added 2026-10-22): outputs/rho_total_limit_sweep_C2_c2_bi40_ss10_class1.csv (lc108 lc109 lr111 lr112 lrc107 lrc108), ..._class2.csv (lc207 lc208 lr210 lr211 lrc207 lrc208),
outputs/rho_total_limit_sweep_C2_c2_bi40_ss10.csv (lr109 class 1, lr208 class 2, first run of the series). The runtime column is not used.

Run: ./venv/bin/python3 scripts/analysis/make_rho_total_limit_sweep_chart.py
Output: figures_export/LiLim Dataset/rho_total_limit_sweep_C2_c2_bi40_ss10.{png,pdf} and _table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import argparse as _ap

_p = _ap.ArgumentParser()
_p.add_argument("--bi", type=int, default=40)  # 2026-10-22: window configuration; default 40/10 = first series (several CSV files), others = one CSV each
_p.add_argument("--ss", type=int, default=10)
_a = _p.parse_args()
BI, SS = _a.bi, _a.ss
TAG = f"bi{BI}_ss{SS}"

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["2", "4", "6", "8", "10", "12", "16", "20", "none"]  # 2026-10-22: L=16 and L=20 added (files ..._L16_20_class1/2.csv)
STEM = "outputs/rho_total_limit_sweep_C2_c2_bi40_ss10"
if (BI, SS) == (40, 10):
    d = pd.concat([pd.read_csv(f"{STEM}_class1.csv"), pd.read_csv(f"{STEM}_class2.csv"), pd.read_csv(f"{STEM}.csv"),
                   pd.read_csv(f"{STEM}_L16_20_class1.csv"), pd.read_csv(f"{STEM}_L16_20_class2.csv")], ignore_index=True)
else:
    d = pd.read_csv(f"outputs/rho_total_limit_sweep_C2_c2_{TAG}.csv")
d["L"] = d["L"].astype(str)
d["cls"] = d["instance"].str.extract(r"^[a-z]+(\d)")[0].map({"1": "Class 1", "2": "Class 2"})
d = d[d.status == "ok"].drop_duplicates(["instance", "L"], keep="last")
d["rate"] = 100 * d["served"] / d["requests"]
COL = {"Class 1": "#276575", "Class 2": "#A6493A", "All instances": "#333333"}  # 2026-10-22: third panel = mean over all 14 instances


def main():
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4), sharey=True)
    for a, cls in zip(ax, ("Class 1", "Class 2", "All instances")):
        s = d if cls == "All instances" else d[d.cls == cls]
        insts = sorted(s.instance.unique())
        for inst in insts:
            r = s[s.instance == inst].set_index("L")["rate"]
            a.plot(range(len(ORDER)), [r[c] for c in ORDER], color=COL[cls], alpha=0.25, linewidth=1.2, zorder=2)
        mean = s.groupby("L")["rate"].mean().reindex(ORDER)
        a.plot(range(len(ORDER)), mean.values, "-o", color=COL[cls], linewidth=2.6, markersize=6, label=f"{cls}: mean of {len(insts)} instances", zorder=3)
        for i, v in enumerate(mean.values):
            a.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color=COL[cls], fontweight="bold")
        mx = s[s.L == "none"]["max_served_by_one_vehicle"]
        a.set_title((f"{cls}: {len(insts)} instances (equal weight per instance)" if cls == "All instances" else f"{cls}: " + ", ".join(insts)) + f"\nwithout limit one vehicle serves {int(mx.min())} to {int(mx.max())} requests", fontsize=9.5)
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(["no limit" if c == "none" else c for c in ORDER])
        a.set_xlabel("Total limit L per vehicle (capacity C=2 on board)", fontsize=10)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(0, 105)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.legend(loc="lower right", fontsize=9, frameon=False)
    ax[0].set_ylabel("Service rate (served / requests)", fontsize=10.5)
    fig.suptitle(f"RHO baseline (no pruner): service rate vs. total requests per vehicle L (C=2 on board, cardinality 2), batch_interval={BI}/step_size={SS}", fontsize=12)
    fig.text(0.5, 0.012, f"Li&Lim, RHO offline (no pruner), max_cardinality=2, batch_interval={BI}, step_size={SS}, 1 run per cell (+-1 request noise); thin lines = single instances;\n"
             f"capacity C=2 via unit-demand manifest copy, L via --max_requests_per_vehicle; sources: outputs/rho_total_limit_sweep_C2_c2_{TAG}*.csv",
             ha="center", fontsize=7.4, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_total_limit_sweep_C2_c2_{TAG}.{ext}", dpi=200)
    plt.close(fig)

    head = " & ".join([f"$L{{=}}{c}$" if c != "none" else "No limit" for c in ORDER])
    lines = []
    for cls in ("Class 1", "Class 2"):
        piv = d[d.cls == cls].pivot(index="instance", columns="L", values="rate")[ORDER]
        lines.append(f"\\multicolumn{{{len(ORDER) + 1}}}{{l}}{{\\textit{{{cls}}}}} \\\\")
        for inst, row in piv.iterrows():
            lines.append(f"{inst} & " + " & ".join(f"{v:.1f}" for v in row.values) + r" \\")
        lines.append("mean & " + " & ".join(f"\\textbf{{{v:.1f}}}" for v in piv.mean().values) + r" \\" + (r" \midrule" if cls == "Class 1" else ""))
    # 2026-10-22: overall mean over all instances (equal weight per instance)
    piv_all = d.pivot(index="instance", columns="L", values="rate")[ORDER]
    lines.append(r"\midrule")
    lines.append(f"mean, all {len(piv_all)} instances & " + " & ".join(f"\\textbf{{{v:.1f}}}" for v in piv_all.mean().values) + r" \\")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
           "\\caption{RHO baseline (no pruner), cardinality 2, vehicle capacity fixed to $C{=}2$ requests on board, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10: service rate (\\%) "
           "as a function of the total limit $L$ of requests per vehicle. One run per cell, $\\pm$1 request noise.}\n"
           "\\begin{tabular}{l" + "c" * len(ORDER) + "}\n\\toprule\nInstance & " + head + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / f"rho_total_limit_sweep_C2_c2_{TAG}_table.tex").write_text(tex)
    print("rows:", len(d), d.status.value_counts().to_dict())
    print(d.groupby(["cls", "L"])["rate"].mean().unstack()[ORDER].round(1).to_string())
    print("all instances:", d.groupby("L")["rate"].mean().reindex(ORDER).round(1).to_dict())
    print(d[d.L == "none"].groupby("cls")["max_served_by_one_vehicle"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
