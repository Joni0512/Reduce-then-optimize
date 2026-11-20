"""
2026-11-19: chart + table for rho_request_pruner_cand_run.py: RHO (offline) 40/10, C=2, cardinality 2, no total limit, 14 Li&Lim test instances (726 requests), 2 reps interleaved, own baseline (no pruner)
vs. Request Pruner WITHOUT (reqbase, 13 features) and WITH (reqplus, 13 + 4 vehicle-related candidate features) the candidate-insertion features; both window-specific for 40/10, trained on the same C=2 data
with the same protocol (sweep_request_pruner_cand.py), threshold 0.1 for both (select_request_pruner_threshold_cand.py, Rule-A analogue on the 8 val instances), min_retention_fraction 0.3.
Speed-up = baseline / variant wall-clock (time.monotonic, sum over 14 instances x 2 reps); kept = requests kept by the pruner / requests seen (sum over windows, includes force-kept active/urgent).
Source: outputs/rho_request_pruner_cand_C2_c2_bi40_ss10.csv
Run: ./venv/bin/python3 scripts/analysis/make_rho_request_pruner_cand_chart.py
Output (figures_export/Request Pruner RHO/): rho_request_pruner_cand_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
SRC = "outputs/rho_request_pruner_cand_C2_c2_bi40_ss10.csv"
d = pd.read_csv(SRC)
d = d[d.status == "ok"].copy()
d["cls"] = d.instance.str.extract(r"^[a-z]+(\d)")[0]
N = 726
V = ["base", "reqbase", "reqplus"]
LAB = {"base": "Baseline\n(no pruner)", "reqbase": "Request Pruner\nwithout vehicle features\n(13 features, t=0.1)", "reqplus": "Request Pruner\nwith vehicle features\n(13 + 4 features, t=0.1)"}
COL = {"base": "#8C8C8C", "reqbase": "#1E2761", "reqplus": "#E8A33D"}
rows = []
pb = d[d.variant == "base"].groupby("instance").served.mean()
for v in V:
    s = d[d.variant == v]
    per = s.groupby("rep").served.sum()
    b = d[d.variant == "base"]
    pv = s.groupby("instance").served.mean()
    r = dict(variant=v, served_mean=per.mean(), rep0=per.iloc[0], rep1=per.iloc[1], rate=100 * per.mean() / N, wall_s=s.wall_s.sum(), speedup=b.wall_s.sum() / s.wall_s.sum(),
             kept_pct=(100 * s.req_kept.sum() / s.req_total.sum()) if s.req_total.sum() > 0 else float("nan"),
             inst_lower=int((pv < pb).sum()) if v != "base" else 0, inst_equal=int((pv == pb).sum()) if v != "base" else 0, inst_higher=int((pv > pb).sum()) if v != "base" else 0)
    for c in ("1", "2"):
        sc, bc = s[s.cls == c], b[b.cls == c]
        r[f"served_c{c}"] = sc.served.sum() / 2
        r[f"speedup_c{c}"] = bc.wall_s.sum() / sc.wall_s.sum()
        r[f"kept_c{c}"] = (100 * sc.req_kept.sum() / sc.req_total.sum()) if sc.req_total.sum() > 0 else float("nan")
    rows.append(r)
T = pd.DataFrame(rows).set_index("variant")
T["diff_req"] = T.served_mean - T.loc["base", "served_mean"]
T["diff_pp"] = 100 * T.diff_req / N
T.round(3).to_csv(OUT / "rho_request_pruner_cand_bi40_ss10_table.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
a = ax[0]
for i, v in enumerate(V):
    a.bar(i, T.loc[v, "rate"], 0.62, color=COL[v])
    a.scatter([i, i], [100 * T.loc[v, "rep0"] / N, 100 * T.loc[v, "rep1"] / N], color="white", edgecolor="black", s=18, zorder=3)
    a.text(i, max(T.loc[v, "rep0"], T.loc[v, "rep1"]) / N * 100 + 0.12, f"{T.loc[v, 'rate']:.1f} %" + ("" if v == "base" else f"\n({T.loc[v, 'diff_pp']:+.2f} pp, {T.loc[v, 'diff_req']:+.1f} req.)"), ha="center", va="bottom", fontsize=8.3)
a.set_xticks(range(len(V)))
a.set_xticklabels([LAB[v] for v in V], fontsize=8)
a.set_ylim(72, 78)
a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
a.set_ylabel("Service rate (mean of 2 reps; dots = single reps)", fontsize=10)
a.set_title("(a) Service rate, 14 instances / 726 requests (y-axis starts at 72 %)", fontsize=10)
b = ax[1]
for i, v in enumerate(V):
    b.bar(i, T.loc[v, "speedup"], 0.62, color=COL[v])
    extra = "" if v == "base" else f"\nkept: {T.loc[v, 'kept_pct']:.0f} % of requests\nclass 1: {T.loc[v, 'speedup_c1']:.2f}x, class 2: {T.loc[v, 'speedup_c2']:.2f}x"
    b.text(i, T.loc[v, "speedup"] + 0.004, f"{T.loc[v, 'speedup']:.2f}x{extra}", ha="center", va="bottom", fontsize=8)
b.axhline(1, color="#444444", lw=0.8, ls=":")
b.set_xticks(range(len(V)))
b.set_xticklabels([LAB[v] for v in V], fontsize=8)
b.set_ylim(0.9, 1.3)
b.set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=10)
b.set_title("(b) Speed-up (14 instances x 2 reps)", fontsize=10)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.suptitle("RHO with the Request Pruner without vs. with vehicle-related features (extra trip time / pickup time), batch_interval=40/step_size=10, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 Li&Lim test instances, 2 reps, local, runs interleaved, own baseline in the same run.\n"
         "Both pruners: window-specific MLP trained on the same C=2 data (36 train/8 val/12 test instances), 432 configs x 5 seeds, winners h64/l3/d0.2/pw1.0/lr0.003 (13 features) and h32/l3/d0.2/pw1.0/lr0.03 (17 features), threshold 0.1 (Rule-A analogue on the 8 val instances),\n"
         f"min_retention_fraction 0.3, active/urgent requests force-kept. Pruned requests are deferred, not rejected. +-1 request noise per instance. Source: {SRC}", ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_request_pruner_cand_bi40_ss10.{ext}", dpi=200)
plt.close(fig)

lines = []
for v in V:
    r = T.loc[v]
    name = {"base": "Baseline (no pruner)", "reqbase": "Request Pruner, 13 features", "reqplus": "Request Pruner, 13 + 4 vehicle features"}[v]
    lines.append(f"{name} & {r.served_mean:.1f} ({r.rate:.1f}) & " + ("--" if v == "base" else f"{r.diff_req:+.1f} ({r.diff_pp:+.2f})") + f" & {r.speedup:.2f}$\\times$ & " + ("--" if v == "base" else f"{r.kept_pct:.0f}") + " \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the window-specific Request Pruner (\\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, threshold 0.1, $C{=}2$, cardinality 2, 14 instances, 726 requests, mean of 2 repetitions) trained without and with the "
       "four vehicle-related features. Speed-up = baseline / variant wall-clock; kept = share of requests passed to trip generation.}\n"
       "\\begin{tabular}{lcccc}\n\\toprule\nVariant & Served (rate \\%) & Diff.\\ to baseline (pp) & Speed-up & Kept (\\%) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_request_pruner_cand_bi40_ss10_table.tex").write_text(tex)
print(T.round(3).to_string())
