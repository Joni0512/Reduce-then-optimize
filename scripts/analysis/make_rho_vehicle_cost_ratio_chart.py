"""
2026-11-19: chart + table for rho_vehicle_cost_ratio_run.py: RHO (offline, no pruner), batch_interval=40, step_size=10, max_cardinality 2, C=2 on board, no total limit, 14 Li&Lim test instances (726 requests),
2 reps interleaved. No-learning vehicle cost-ratio filter (per non-active request keep only vehicles with added_cost <= rho x added_cost of the best vehicle; best always kept) with
rho in {3 (user request), 2, 1.5} from run 1 and rho = 1.1 from run 2. EACH run measured its OWN baseline in the same job, so every variant is compared to the baseline of its own run
(the two baselines differ by RHO noise: 548.0 vs 548.5 served per rep). Speed-up = baseline / variant wall-clock (sum over 14 instances x 2 reps, time.monotonic of the solver subprocess);
kept = share of single-trip costs kept by the filter.
Excluded from the RTV-time sums: lrc208 rep 0 (both variants of run 2): its log reports one window with 192.5 s "feasible trip costs generated" although the whole subprocess took 16 s
(wall-clock is time.monotonic) -> the solver's time.time()-based log value is corrupt (probably a system clock jump); wall-clock and served requests of that run are unaffected and kept.
Sources: outputs/rho_vehicle_cost_ratio_C2_c2_bi40_ss10.csv (rho 3, 2, 1.5), outputs/rho_vehicle_cost_ratio_C2_c2_bi40_ss10_rho1.1.csv (rho 1.1)
Run: ./venv/bin/python3 scripts/analysis/make_rho_vehicle_cost_ratio_chart.py
Output (figures_export/Request Pruner RHO/): rho_vehicle_cost_ratio_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
SRC = ["outputs/rho_vehicle_cost_ratio_C2_c2_bi40_ss10.csv", "outputs/rho_vehicle_cost_ratio_C2_c2_bi40_ss10_rho1.1.csv"]
N = 726
BAD_RTV = {("lrc208", 0)}  # (instance, rep) with corrupt solver-log time, see docstring; excluded from RTV sums of BOTH variants of that run
VARS = ["vcr3", "vcr2", "vcr1.5", "vcr1.1"]
LAB = {"vcr3": "rho = 3", "vcr2": "rho = 2", "vcr1.5": "rho = 1.5", "vcr1.1": "rho = 1.1"}
COL = {"vcr3": "#1E2761", "vcr2": "#2A9D8F", "vcr1.5": "#E8A33D", "vcr1.1": "#B5473A"}
rows, bases = [], []
for src in SRC:
    d = pd.read_csv(src)
    d = d[d.status == "ok"]
    rtv_ok = d[[(i, r) not in BAD_RTV for i, r in zip(d.instance, d.rep)]]
    base = d[d.variant == "base"]
    bper = base.groupby("rep").served.sum()
    bases.append(dict(source=src, served_per_rep=bper.mean(), rate=100 * bper.mean() / N, wall_s=base.wall_s.sum(), rtv_s=rtv_ok[rtv_ok.variant == "base"].rtv_s.sum()))
    pb = d[d.variant == "base"].groupby("instance").served.mean()
    for v in [x for x in d.variant.unique() if x != "base"]:
        s = d[d.variant == v]
        per = s.groupby("rep").served.sum()
        pv = s.groupby("instance").served.mean()
        rows.append(dict(variant=v, served_mean=per.mean(), rep0=per.iloc[0], rep1=per.iloc[1], rate=100 * per.mean() / N, base_served=bper.mean(), base_rate=100 * bper.mean() / N,
                         diff_req=per.mean() - bper.mean(), diff_pp=100 * (per.mean() - bper.mean()) / N,
                         speedup_wall=base.wall_s.sum() / s.wall_s.sum(), speedup_rtv=rtv_ok[rtv_ok.variant == "base"].rtv_s.sum() / rtv_ok[rtv_ok.variant == v].rtv_s.sum(),
                         kept_pct=100 * s.single_costs_kept.sum() / s.single_costs_total.sum(),
                         inst_lower=int((pv < pb).sum()), inst_equal=int((pv == pb).sum()), inst_higher=int((pv > pb).sum())))
T = pd.DataFrame(rows).set_index("variant").reindex(VARS)
T.round(3).to_csv(OUT / "rho_vehicle_cost_ratio_bi40_ss10_table.csv")
pd.DataFrame(bases).round(2).to_csv(OUT / "rho_vehicle_cost_ratio_bi40_ss10_baselines.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.8))
a = ax[0]
for i, v in enumerate(VARS):
    a.bar(i, T.loc[v, "diff_pp"], 0.65, color=COL[v])
    a.scatter([i, i], [100 * (T.loc[v, "rep0"] - T.loc[v, "base_served"]) / N, 100 * (T.loc[v, "rep1"] - T.loc[v, "base_served"]) / N], color="white", edgecolor="black", s=18, zorder=3)
    y = T.loc[v, "diff_pp"]
    a.text(i, max(y, 100 * (max(T.loc[v, "rep0"], T.loc[v, "rep1"]) - T.loc[v, "base_served"]) / N) + 0.04, f"{y:+.2f} pp\n({T.loc[v, 'diff_req']:+.1f} req.)\n{T.loc[v, 'rate']:.1f} % vs. {T.loc[v, 'base_rate']:.1f} %", ha="center", va="bottom", fontsize=7.8)
a.axhline(0, color="#444444", lw=0.8)
a.set_xticks(range(len(VARS)))
a.set_xticklabels([LAB[v] for v in VARS])
a.set_ylim(-1.0, 1.0)
a.set_ylabel("Service rate vs. own baseline (pp; mean of 2 reps, dots = single reps)", fontsize=9.5)
a.set_title("(a) Change of the service rate, 14 instances / 726 requests", fontsize=10)
b = ax[1]
for i, v in enumerate(VARS):
    b.bar(i, T.loc[v, "speedup_wall"], 0.65, color=COL[v])
    b.text(i, T.loc[v, "speedup_wall"] + 0.003, f"{T.loc[v, 'speedup_wall']:.2f}x\nkept: {T.loc[v, 'kept_pct']:.0f} %", ha="center", va="bottom", fontsize=8.5)
b.axhline(1, color="#444444", lw=0.8, ls=":")
b.set_xticks(range(len(VARS)))
b.set_xticklabels([LAB[v] for v in VARS])
b.set_ylim(0.9, 1.1)
b.set_ylabel("Speed-up = own baseline wall-clock / variant wall-clock", fontsize=10)
b.set_title("(b) Speed-up (14 instances x 2 reps); kept = share of single-trip costs kept", fontsize=10)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.suptitle("RHO with the vehicle cost-ratio filter (keep vehicle only if added_cost <= rho x best vehicle's), batch_interval=40/step_size=10, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, no pruner, 14 Li&Lim test instances, 2 reps, local, runs interleaved.\n"
         "Filter: per non-active request keep vehicles with added_cost <= rho x added_cost of the best vehicle, on the single-trip costs before the shared trips (TripHandler._filter_vehicle_edges_by_cost_ratio).\n"
         "rho=3 requested by the user, 2 and 1.5 added (run 1), 1.1 requested later (run 2); each run has its own baseline (548.0 and 548.5 served per rep). +-1 request noise per instance.\n"
         f"Sources: {SRC[0]}; {SRC[1]}", ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_vehicle_cost_ratio_bi40_ss10.{ext}", dpi=200)
plt.close(fig)

lines = []
for v in VARS:
    r = T.loc[v]
    name = LAB[v].replace("rho = ", "$\\rho{=}$")  # outside any f-string (py3.11: no backslash in f-string expressions)
    lines.append(f"{name} & {r.served_mean:.1f} ({r.rate:.1f}) & {r.base_served:.1f} ({r.base_rate:.1f}) & {r.diff_req:+.1f} ({r.diff_pp:+.2f}) & {r.speedup_wall:.2f}$\\times$ & {r.kept_pct:.0f} \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the vehicle cost-ratio filter (a vehicle is kept for a request only if its added cost is at most $\\rho$ times that of the best vehicle), \\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, "
       "$C{=}2$, cardinality 2, 14 instances (726 requests), mean of 2 repetitions. Each row is compared with the baseline of its own run. Speed-up = baseline / variant wall-clock; kept = share of single-trip costs kept.}\n"
       "\\begin{tabular}{lcccrr}\n\\toprule\nVariant & Served (rate \\%) & Own baseline & Diff.\\ (pp) & Speed-up & Kept (\\%) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_vehicle_cost_ratio_bi40_ss10_table.tex").write_text(tex)
print(T.round(3).to_string())
