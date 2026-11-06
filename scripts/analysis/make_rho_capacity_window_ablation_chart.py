"""
2026-10-22: ablation of the window configuration for the CAPACITY sweep: RHO baseline service rate vs. the vehicle capacity C (requests on board at the same time,
unit-demand manifest copy am=1, am_capacity=C; "orig" = original manifest), Li&Lim, cardinality 2, no total limit, no pruner, one run per cell, for four
(batch_interval, step_size) pairs: 40/10 (first series), 100/25, 120/30, 200/50. 14 instances (7 class 1, 7 class 2).
Timeouts (30 min per run) have no service rate and are left out of the means (the number of instances behind each mean is printed).

Sources: 40/10: outputs/rho_capacity_sweep_c2_bi40_ss10.csv (lr109, lr208), ..._class1.csv, ..._class2.csv; 100/25, 120/30, 200/50: outputs/rho_capacity_sweep_c2_bi100_ss25.csv,
..._bi120_ss30.csv, ..._bi200_ss50.csv. RHO is not bit-deterministic: +-1 request (about 2 pp) per cell is noise. The runtime column of the CSVs is not used.

Run: ./venv/bin/python3 scripts/analysis/make_rho_capacity_window_ablation_chart.py
Output (figures_export/LiLim Dataset/): rho_capacity_window_ablation_c2.{png,pdf} (means), ..._per_instance.{png,pdf}, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["orig", "2", "3", "4", "6", "7", "10", "12", "20", "160"]
P = "outputs/rho_capacity_sweep_c2_"
CONFIGS = {
    "batch_interval=40/step_size=10": [P + s for s in ("bi40_ss10.csv", "bi40_ss10_class1.csv", "bi40_ss10_class2.csv")],
    "batch_interval=100/step_size=25": [P + "bi100_ss25.csv"],
    "batch_interval=120/step_size=30": [P + "bi120_ss30.csv"],
    "batch_interval=200/step_size=50": [P + "bi200_ss50.csv"],
}
COL = {"batch_interval=40/step_size=10": "#1E2761", "batch_interval=100/step_size=25": "#E8A33D", "batch_interval=120/step_size=30": "#3E8F6C", "batch_interval=200/step_size=50": "#A6493A"}

frames = []
for name, files in CONFIGS.items():
    x = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    x["config"] = name
    frames.append(x)
d = pd.concat(frames, ignore_index=True)
d["C"] = d["C"].astype(str)
d["cls"] = d["instance"].str.extract(r"^[a-z]+(\d)")[0].map({"1": "Class 1", "2": "Class 2"})
n_timeout = d[d.status != "ok"].groupby("config").size().to_dict()
d = d[d.status == "ok"].drop_duplicates(["config", "instance", "C"], keep="last")
d["rate"] = 100 * d["served"] / d["requests"]


def main():
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), sharey=True)
    for a, cls in zip(ax, ("Class 1", "Class 2", "All instances")):
        s = d if cls == "All instances" else d[d.cls == cls]
        for name in CONFIGS:
            m = s[s.config == name].groupby("C")["rate"].mean().reindex(ORDER)
            a.plot(range(len(ORDER)), m.values, "-o", color=COL[name], linewidth=2.2, markersize=5, label=name)
        a.set_title(f"{cls}: mean over {s.instance.nunique()} instances", fontsize=11)
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(["orig" if c == "orig" else c for c in ORDER])
        a.set_xlabel("Vehicle capacity C (requests on board at the same time; orig = original capacity)", fontsize=10)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(40, 100)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("Service rate (served / requests)", fontsize=10.5)
    ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    to = ", ".join(f"{k.split('=')[1].split('/')[0]}/{k.split('=')[-1]}: {v}" for k, v in n_timeout.items()) or "none"
    fig.suptitle("RHO baseline (no pruner): service rate vs. vehicle capacity C for four window configurations (cardinality 2)", fontsize=12)
    fig.text(0.5, 0.012, "Li&Lim, RHO offline, max_cardinality=2, 1 run per cell (+-1 request noise = about 2 pp), equal weight per instance, timeout cells (30 min) left out [" + to + "]; capacity via unit-demand manifest copy;\n"
             "sources: outputs/rho_capacity_sweep_c2_bi40_ss10*.csv, ..._bi100_ss25.csv, ..._bi120_ss30.csv, ..._bi200_ss50.csv", ha="center", fontsize=7.4, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_capacity_window_ablation_c2.{ext}", dpi=200)
    plt.close(fig)

    insts = sorted(d.instance.unique(), key=lambda s: (int([c for c in s if c.isdigit()][0]), s))
    fig, axes = plt.subplots(4, 4, figsize=(16, 12.5), sharey=True)
    for a in axes.flat:
        a.set_visible(False)
    for a, inst in zip(axes.flat, insts):
        a.set_visible(True)
        n_req = int(d[d.instance == inst].requests.iloc[0])
        for name in CONFIGS:
            r = d[(d.instance == inst) & (d.config == name)].set_index("C")["rate"].reindex(ORDER)
            a.plot(range(len(ORDER)), r.values, "-o", color=COL[name], linewidth=1.6, markersize=3.5, label=name)
        a.set_title(f"{inst} ({n_req} requests)", fontsize=9.5)
        a.set_xticks(range(len(ORDER)))
        a.set_xticklabels(["o", "2", "3", "4", "6", "7", "10", "12", "20", "160"], fontsize=7.5)
        a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        a.set_ylim(30, 105)
        a.grid(True, linestyle=":", alpha=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    for a in axes[-1, :]:
        if a.get_visible():
            a.set_xlabel("Vehicle capacity C (o = original)", fontsize=9)
    fig.suptitle("RHO baseline (no pruner): service rate per instance vs. vehicle capacity C for four window configurations (cardinality 2)", fontsize=12)
    fig.text(0.5, 0.008, "Li&Lim, RHO offline, max_cardinality=2, 1 run per cell (+-1 request noise = about 2 pp); missing points = timeout (30 min); capacity via unit-demand manifest copy;\n"
             "sources: outputs/rho_capacity_sweep_c2_bi40_ss10*.csv, ..._bi100_ss25.csv, ..._bi120_ss30.csv, ..._bi200_ss50.csv", ha="center", fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rho_capacity_window_ablation_c2_per_instance.{ext}", dpi=170)
    plt.close(fig)

    head = " & ".join(["Original"] + [f"$C{{=}}{c}$" for c in ORDER[1:]])
    lines = []
    for block, s in (("All instances", d), ("Class 1 (7 instances)", d[d.cls == "Class 1"]), ("Class 2 (7 instances)", d[d.cls == "Class 2"])):
        lines.append(f"\\multicolumn{{{len(ORDER) + 1}}}{{l}}{{\\textit{{{block}}}}} \\\\")
        for name in CONFIGS:
            m = s[s.config == name].groupby("C")["rate"].mean().reindex(ORDER)
            bi, ss = name.replace("batch_interval=", "").split("/step_size=")
            lines.append(f"{bi}/{ss} & " + " & ".join(f"{v:.1f}" for v in m.values) + r" \\")
        if block != "Class 2 (7 instances)":
            lines.append(r"\midrule")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
           "\\caption{RHO baseline (no pruner), cardinality 2: mean service rate (\\%) over instances as a function of the vehicle capacity $C$ (requests on board at the same time, unit demand; ``Original'' = capacity 200/1000 "
           "with real demands) for four window configurations (\\texttt{batch\\_interval}/\\texttt{step\\_size}). One run per cell, $\\pm$1 request noise, equal weight per instance; timeout cells (30 min) are left out of the means.}\n"
           "\\begin{tabular}{l" + "c" * len(ORDER) + "}\n\\toprule\nWindow & " + head + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / "rho_capacity_window_ablation_c2_table.tex").write_text(tex)

    print("cells per config:", d.groupby("config").size().to_dict(), "| timeouts:", n_timeout)
    for block, s in (("ALL", d), ("Class 1", d[d.cls == "Class 1"]), ("Class 2", d[d.cls == "Class 2"])):
        print(f"\n{block}:")
        print(s.groupby(["config", "C"])["rate"].mean().unstack()[ORDER].round(1).to_string())


if __name__ == "__main__":
    main()
