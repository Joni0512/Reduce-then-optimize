"""
2026-11-11: chart + table for rho_extra_vehicles.py: RHO (offline, no pruner), batch_interval=40, step_size=10, max_cardinality 2, C=2 on board, no total limit, 14 Li&Lim test instances (726 requests),
original fleet (89 vehicles) vs. 100 vehicles in total (11 extra vehicles, only class 2, round-robin lc207 lc208 lr208 lr210 lr211 lrc207 lrc208 -> first 4 get +2, last 3 get +1), 2 reps, runs interleaved.
Source: outputs/rho_extra_vehicles_C2_c2_bi40_ss10.csv. Class 1 has the SAME fleet in both variants -> its base/extra difference is the run-to-run noise.
Run: ./venv/bin/python3 scripts/analysis/make_rho_extra_vehicles_chart.py
Output (figures_export/LiLim Dataset/): rho_extra_vehicles_100_C2_c2_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
SRC = "outputs/rho_extra_vehicles_C2_c2_bi40_ss10.csv"
d = pd.read_csv(SRC)
d = d[d.status == "ok"].copy()
d["cls"] = d.instance.str.extract(r"^[a-z]+(\d)")[0]
CB, CP = "#8C8C8C", "#E8A33D"

inst = d.groupby(["instance", "variant"]).agg(served=("served", "mean"), veh=("vehicles", "first"), req=("requests", "first"), sec=("seconds", "sum")).unstack()
cls_rows = []
for name, sel in (("Class 1 (7 inst., 371 req.)", "1"), ("Class 2 (7 inst., 355 req.)", "2"), ("All (14 inst., 726 req.)", None)):
    x = d if sel is None else d[d.cls == sel]
    n_req = x[x.variant == "base"].groupby("rep").requests.sum().iloc[0]
    row = dict(group=name, requests=int(n_req))
    for v in ("base", "extra"):
        per_rep = x[x.variant == v].groupby("rep").served.sum()
        row[f"{v}_served"] = per_rep.mean()
        row[f"{v}_rate"] = 100 * per_rep.mean() / n_req
        row[f"{v}_vehicles"] = int(x[(x.variant == v) & (x.rep == 0)].vehicles.sum())
        row[f"{v}_sec"] = x[x.variant == v].seconds.sum()
    row["diff_req"] = row["extra_served"] - row["base_served"]
    row["diff_pp"] = row["extra_rate"] - row["base_rate"]
    cls_rows.append(row)
T = pd.DataFrame(cls_rows)
T.round(2).to_csv(OUT / "rho_extra_vehicles_100_C2_c2_bi40_ss10_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.9), gridspec_kw=dict(width_ratios=[1, 1.7]))
a = ax[0]
for i, r in T.iterrows():
    for k, (v, c) in enumerate((("base", CB), ("extra", CP))):
        x = i + (k - 0.5) * 0.38
        a.bar(x, r[f"{v}_rate"], 0.38, color=c, label={"base": "Original fleet", "extra": "+11 vehicles (class 2 only)"}[v] if i == 0 else None)
        a.text(x, r[f"{v}_rate"] + 0.6, f"{r[f'{v}_rate']:.1f} %\n({r[f'{v}_vehicles']} veh.)", ha="center", va="bottom", fontsize=7.8)
a.set_xticks(range(len(T)))
a.set_xticklabels([g.replace(" (", "\n(") for g in T.group], fontsize=8.5)
a.set_ylim(50, 100)
a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
a.set_ylabel("Service rate (mean of 2 reps)", fontsize=10)
a.set_title("(a) By class (y-axis starts at 50 %)", fontsize=10)
a.legend(loc="upper left", frameon=False, fontsize=8.5)
b = ax[1]
c2 = [s for s in inst.index if s[:3] in ("lc2", "lr2", "lrc") and s[-3] == "2" or s in ("lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208")]
c2 = ["lc207", "lc208", "lr208", "lr210", "lr211", "lrc207", "lrc208"]
for i, s in enumerate(c2):
    for k, (v, c) in enumerate((("base", CB), ("extra", CP))):
        x = i + (k - 0.5) * 0.38
        y = inst.loc[s, ("served", v)]
        b.bar(x, y, 0.38, color=c)
        b.text(x, y + 0.4, f"{y:.1f}\n({int(inst.loc[s, ('veh', v)])} veh.)", ha="center", va="bottom", fontsize=7.4)
b.set_xticks(range(len(c2)))
b.set_xticklabels([f"{s}\n(+{int(inst.loc[s, ('veh', 'extra')] - inst.loc[s, ('veh', 'base')])} veh., {int(inst.loc[s, ('req', 'base')])} req.)" for s in c2], fontsize=8)
b.set_ylim(0, 60)
b.set_ylabel("Served requests (mean of 2 reps)", fontsize=10)
b.set_title("(b) Class-2 instances, served requests (in brackets: vehicles of the fleet)", fontsize=10)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.suptitle("RHO (offline, no pruner): original fleet (89 vehicles) vs. 100 vehicles in total, batch_interval=40/step_size=10, C=2, max_cardinality=2, no total limit", fontsize=11.5)
fig.text(0.5, 0.012,
         "Config: RHO --mode offline, batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no --max_requests_per_vehicle, no pruner, 14 Li&Lim test instances (726 requests), 2 reps, local, runs interleaved (base, extra).\n"
         "Extra vehicles: copies of the state of vehicle 0 (depot, start/end time, capacity) with empty manifest, only class 2, round-robin lc207 lc208 lr208 lr210 lr211 lrc207 lrc208 (first 4: +2, last 3: +1). "
         "Class 1 has the same fleet in both variants: its difference is the run-to-run noise.\n"
         f"Source: {SRC}",
         ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_extra_vehicles_100_C2_c2_bi40_ss10.{ext}", dpi=200)
plt.close(fig)

lines = []
for _, r in T.iterrows():
    lines.append(f"{r.group.replace('&', chr(92) + '&')} & {int(r.base_vehicles)} & {r.base_served:.1f} ({r.base_rate:.1f}) & {int(r.extra_vehicles)} & {r.extra_served:.1f} ({r.extra_rate:.1f}) & {r.diff_req:+.1f} ({r.diff_pp:+.1f}) \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO without pruner, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, $C{=}2$, cardinality 2, no total limit: original fleet vs.\\ 100 vehicles in total (11 extra vehicles, class 2 only). "
       "Served requests (service rate in \\%), mean of 2 repetitions. Class 1 has the same fleet in both variants (difference = noise).}\n"
       "\\begin{tabular}{lrcrcc}\n\\toprule\n & \\multicolumn{2}{c}{Original fleet} & \\multicolumn{2}{c}{100 vehicles} & \\\\\n"
       "Group & Veh. & Served (rate \\%) & Veh. & Served (rate \\%) & Difference (pp) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_extra_vehicles_100_C2_c2_bi40_ss10_table.tex").write_text(tex)
print(T.round(2).to_string(index=False))
