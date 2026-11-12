"""
2026-11-09: RHO service rate vs. vehicle capacity C (persons on board at the same time), max_cardinality 2 vs. 3, batch_interval=40, step_size=10, NO total limit per vehicle,
no pruner, Li&Lim, 1 run per cell. Only the 12 instances that finished for EVERY C with cardinality 3 (lr208 and lrc208 are excluded: timeout also with the 2 h limit for C>=4/5),
so every bar covers the same 625 requests (7 class-1 instances = 371 requests: lc108 lc109 lr109 lr111 lr112 lrc107 lrc108; 5 class-2 instances = 254 requests: lc207 lc208 lr210 lr211 lrc207).

Sources:
  cardinality 3: outputs/rho_total_limit_sweep_C{2..6}_c3_bi40_ss10.csv (+ ..._retry2h.csv, 2 h per-instance limit, replaces the timeouts of the first 30-min run)
  cardinality 2: outputs/rho_capacity_sweep_c2_bi40_ss10.csv (lr109, lr208), ..._class1.csv, ..._class2.csv for C=2,3,4,6; C=5 from outputs/rho_total_limit_sweep_C5_c2_bi40_ss10.csv
RHO is not bit-deterministic: +-1 request per instance is noise (12 instances: a few requests = about 0.5 pp). Runtime is NOT compared (parallel runs).

Run: ./venv/bin/python3 scripts/analysis/make_rho_capacity_cardinality_chart.py
Output (figures_export/LiLim Dataset/): rho_capacity_cardinality_bi40_ss10_12inst.{png,pdf}, ..._table.tex, ..._table.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/LiLim Dataset")
OUT.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"lr208", "lrc208"}
CS = [2, 3, 4, 5, 6]


def card3(C):
    d = pd.read_csv(f"outputs/rho_total_limit_sweep_C{C}_c3_bi40_ss10.csv")
    p = Path(f"outputs/rho_total_limit_sweep_C{C}_c3_bi40_ss10_retry2h.csv")
    if p.exists():
        r = pd.read_csv(p)
        r = r[r.status == "ok"]
        d = pd.concat([d[~d.instance.isin(r.instance)], r])
    return d[d.status == "ok"].drop_duplicates("instance", keep="last")


def card2(C):
    if C == 5:
        d = pd.read_csv("outputs/rho_total_limit_sweep_C5_c2_bi40_ss10.csv")
    else:
        d = pd.concat([pd.read_csv(f"outputs/rho_capacity_sweep_c2_bi40_ss10{s}.csv") for s in ("", "_class1", "_class2")])
        d = d[d.C.astype(str) == str(C)]
    return d[d.status == "ok"].drop_duplicates("instance", keep="last")


rows = []
for C in CS:
    for card, fn in ((2, card2), (3, card3)):
        d = fn(C)
        d = d[~d.instance.isin(EXCLUDE)]
        assert len(d) == 12, (C, card, len(d))
        d = d.assign(cls=d.instance.str.extract(r"^[a-z]+(\d)")[0])
        rows.append(dict(C=C, card=card, served=int(d.served.sum()), requests=int(d.requests.sum()),
                         c1=int(d[d.cls == "1"].served.sum()), c1n=int(d[d.cls == "1"].requests.sum()),
                         c2=int(d[d.cls == "2"].served.sum()), c2n=int(d[d.cls == "2"].requests.sum())))
t = pd.DataFrame(rows)
t["rate"] = 100 * t.served / t.requests
t.to_csv(OUT / "rho_capacity_cardinality_bi40_ss10_12inst_table.csv", index=False)

CB, CP = "#1E2761", "#E8A33D"
fig, ax = plt.subplots(figsize=(10.5, 5.8))
w = 0.38
for k, (card, col) in enumerate(((2, CB), (3, CP))):
    s = t[t.card == card].set_index("C").reindex(CS)
    xs = [i + (k - 0.5) * w for i in range(len(CS))]
    bars = ax.bar(xs, s.rate.values, w, color=col, label=f"max_cardinality={card}")
    for x, r, n in zip(xs, s.rate.values, s.served.values):
        ax.text(x, r + 0.4, f"{r:.1f} %\n({n}/625)", ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(range(len(CS)))
ax.set_xticklabels([f"C={c}" for c in CS])
ax.set_xlabel("Vehicle capacity C (persons on board at the same time)", fontsize=10.5)
ax.set_ylabel("Service rate (served / requests)", fontsize=10.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(60, 100)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)  # 2026-11-09: was lower right, overlapped the bars
fig.suptitle("RHO (offline, no pruner): service rate vs. vehicle capacity C, batch_interval=40/step_size=10, no total limit per vehicle", fontsize=11.5)
ax.set_title("12 Li&Lim instances (625 requests; lr208 and lrc208 excluded: timeout even at 2 h for max_cardinality=3, C>=4), y-axis starts at 60 %", fontsize=9)
fig.text(0.5, 0.012, "Config: RHO --mode offline, batch_interval=40, step_size=10, no --max_requests_per_vehicle, C via unit-demand manifest copy, 1 run per cell (+-1 request noise per instance).\n"
         "Instances: lc108 lc109 lr109 lr111 lr112 lrc107 lrc108 (class 1, 371 requests), lc207 lc208 lr210 lr211 lrc207 (class 2, 254 requests).\n"
         "Sources: cardinality 3 = outputs/rho_total_limit_sweep_C{2..6}_c3_bi40_ss10(_retry2h).csv; cardinality 2 = outputs/rho_capacity_sweep_c2_bi40_ss10(_class1/_class2).csv\n"
         "(C=2,3,4,6) and outputs/rho_total_limit_sweep_C5_c2_bi40_ss10.csv (C=5).",
         ha="center", fontsize=7, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_capacity_cardinality_bi40_ss10_12inst.{ext}", dpi=200)
plt.close(fig)

lines = []
for C in CS:
    a, b = t[(t.C == C) & (t.card == 2)].iloc[0], t[(t.C == C) & (t.card == 3)].iloc[0]
    sa, sb = int(a.served), int(b.served)  # int(): iloc[0] of a mixed-dtype row yields floats
    lines.append(f"$C{{=}}{C}$ & {sa} ({a.rate:.1f}) & {sb} ({b.rate:.1f}) & {sb - sa:+d} ({b.rate - a.rate:+.1f}) \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO without pruner, \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, no total limit per vehicle: served requests (service rate in \\%) vs.\\ vehicle capacity $C$ "
       "and \\texttt{max\\_cardinality}. 12 Li\\&Lim instances (625 requests), \\texttt{lr208} and \\texttt{lrc208} excluded (timeout for cardinality 3, $C\\geq 4$). One run per cell, $\\pm$1 request noise per instance.}\n"
       "\\begin{tabular}{lccc}\n\\toprule\nCapacity & Cardinality 2 & Cardinality 3 & Difference \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_capacity_cardinality_bi40_ss10_12inst_table.tex").write_text(tex)
print(t.to_string(index=False))
