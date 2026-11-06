"""
2026-10-22: RHO baseline service rate as a function of the vehicle capacity C (in requests on board at the same time), Li&Lim, cardinality 2.

Setup (all runs): RHO offline, no pruner, --max_cardinality 2, --batch_interval 40, --step_size 10, one run per cell, seed default; capacity C is
set by a COPY of the manifest with am=1 for every request and am_capacity=C for every vehicle (scripts/analysis/rho_capacity_sweep.py); "orig" =
original manifest (capacity 200 / 1000, real demands). Fleet = number of routes of the reference solution (li_lim_parser), manifests are cleared before the run.
RHO is not bit-deterministic: +-1 request (about 2 pp) per cell is noise.
Sources: outputs/rho_capacity_sweep_c2_bi40_ss10_class1.csv (lc108 lc109 lr111 lr112 lrc107 lrc108), ..._class2.csv (lc207 lc208 lr210 lr211 lrc207 lrc208),
outputs/rho_capacity_sweep_c2_bi40_ss10.csv (lr109 class 1, lr208 class 2; same setup, first run of the series).
The runtime column of these CSVs is NOT used (laptop sleep + parallel runs distorted it).

Run: ./venv/bin/python3 scripts/analysis/make_rho_capacity_sweep_chart.py
Output: figures_export/LiLim Dataset/rho_capacity_sweep_c2_bi40_ss10.{png,pdf} and _table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["2", "3", "4", "6", "7", "10", "12", "20", "160"]

parts = [pd.read_csv("outputs/rho_capacity_sweep_c2_bi40_ss10_class1.csv"), pd.read_csv("outputs/rho_capacity_sweep_c2_bi40_ss10_class2.csv"),
         pd.read_csv("outputs/rho_capacity_sweep_c2_bi40_ss10.csv")]
d = pd.concat(parts, ignore_index=True)
d["C"] = d["C"].astype(str)
d["cls"] = d["instance"].str.extract(r"^[a-z]+(\d)")[0].map({"1": "Class 1", "2": "Class 2"})
d = d[d.status == "ok"].drop_duplicates(["instance", "C"], keep="last")
d["rate"] = 100 * d["served"] / d["requests"]

COL = {"Class 1": "#276575", "Class 2": "#A6493A", "All instances": "#333333"}  # 2026-10-22: third panel = mean over all instances


def main():
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4), sharey=True)
    for a, cls in zip(ax, ("Class 1", "Class 2", "All instances")):
        s = d if cls == "All instances" else d[d.cls == cls]
        insts = sorted(s.instance.unique())
        for inst in insts:
            r = s[s.instance == inst].set_index("C")["rate"]
            a.plot(range(len(ORDER)), [r[c] for c in ORDER], color=COL[cls], alpha=0.25, linewidth=1.2, zorder=2)
        mean = s[s.C.isin(ORDER)].groupby("C")["rate"].mean().reindex(ORDER)
        a.plot(range(len(ORDER)), mean.values, "-o", color=COL[cls], linewidth=2.6, markersize=6, label=f"{cls}: mean of {len(insts)} instances", zorder=3)
        orig = s[s.C == "orig"]["rate"].mean()
        a.axhline(orig, color="#444444", linestyle="--", linewidth=1.3, zorder=1)
        a.annotate(f"original capacity (mean {orig:.1f} %)", (len(ORDER) - 1, orig), textcoords="offset points", xytext=(0, 5), ha="right", fontsize=8.5, color="#444444")
        for i, v in enumerate(mean.values):
            a.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color=COL[cls], fontweight="bold")
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(ORDER)
        a.set_xlabel("Vehicle capacity C (requests on board at the same time)", fontsize=10.5)
        a.set_title(f"{cls}: {len(insts)} instances (equal weight per instance)" if cls == "All instances" else f"{cls}: " + ", ".join(insts), fontsize=10)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(45, 105)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.legend(loc="lower right", fontsize=9, frameon=False)
    ax[0].set_ylabel("Service rate (served / requests)", fontsize=10.5)
    fig.suptitle("RHO baseline (no pruner): service rate vs. vehicle capacity C, cardinality 2", fontsize=12.5)
    fig.text(0.5, 0.012, "Li&Lim, RHO offline (no pruner), max_cardinality=2, batch_interval=40, step_size=10, 1 run per cell (+-1 request noise); thin lines = single instances;\n"
             "capacity via unit-demand manifest copy (am=1, am_capacity=C); sources: outputs/rho_capacity_sweep_c2_bi40_ss10.csv, ..._class1.csv, ..._class2.csv",
             ha="center", fontsize=7.4, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_capacity_sweep_c2_bi40_ss10.{ext}", dpi=200)
    plt.close(fig)

    # LaTeX table (English): mean service rate per class and C, plus per-instance rows
    cols = ["orig"] + ORDER
    head = " & ".join(["Original"] + [f"$C{{=}}{c}$" for c in ORDER])
    lines = []
    for cls in ("Class 1", "Class 2"):
        s = d[d.cls == cls]
        piv = s.pivot(index="instance", columns="C", values="rate")[cols]
        lines.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{cls}}}}} \\\\")
        for inst, row in piv.iterrows():
            lines.append(f"{inst} & " + " & ".join(f"{v:.1f}" for v in row.values) + r" \\")
        lines.append("mean & " + " & ".join(f"\\textbf{{{v:.1f}}}" for v in piv.mean().values) + r" \\" + (r" \midrule" if cls == "Class 1" else ""))
    # 2026-10-22: overall mean over all instances (equal weight per instance)
    piv_all = d.pivot(index="instance", columns="C", values="rate")[cols]
    lines.append(r"\midrule")
    lines.append(f"mean, all {len(piv_all)} instances & " + " & ".join(f"\\textbf{{{v:.1f}}}" for v in piv_all.mean().values) + r" \\")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
           "\\caption{RHO baseline (no pruner), cardinality 2, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10: service rate (\\%) as a function of the vehicle capacity $C$ "
           "(requests on board at the same time, unit demand). One run per cell, $\\pm$1 request noise. ``Original'' = capacity 200/1000 with the real demands.}\n"
           "\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nInstance & " + head + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / "rho_capacity_sweep_c2_bi40_ss10_table.tex").write_text(tex)
    print("saved", OUT / "rho_capacity_sweep_c2_bi40_ss10.png", "/ .pdf / _table.tex")
    print(d.groupby(["cls", "C"])["rate"].mean().unstack()[cols].round(1).to_string())
    print("all instances:", d.groupby("C")["rate"].mean().reindex(cols).round(1).to_dict())


if __name__ == "__main__":
    main()
