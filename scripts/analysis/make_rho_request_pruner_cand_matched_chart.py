"""
2026-11-20: chart + table for the FAIR comparison of the Request Pruner trained without (13 features) and with (13 + 4 vehicle-related candidate features) the candidate-insertion features at a
MATCHED pruning rate. rho_request_pruner_cand_run.py with variants base, reqbase@0.1, reqplus@0.02, reqbase@0.2, reqplus@0.1: the thresholds were picked on the 8 val instances
(select_request_pruner_threshold_cand.py --ths ...) so that the two pruners prune the same share of candidate rows: pair 1 = 36.3 % (t=0.1, 13 features) vs 35.4 % (t=0.02, 17 features),
pair 2 = 49.6 % (t=0.2, 13 features) vs 50.9 % (t=0.1, 17 features). RHO offline, batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit,
14 Li&Lim test instances (726 requests), 2 reps interleaved, own baseline in the same run, min_retention_fraction 0.3.
Speed-up = baseline / variant wall-clock (time.monotonic, sum over 14 instances x 2 reps); kept = requests passed to trip generation / requests seen (sum over windows, incl. force-kept).
Source: outputs/rho_request_pruner_cand_C2_c2_bi40_ss10_matched.csv
Run: ./venv/bin/python3 scripts/analysis/make_rho_request_pruner_cand_matched_chart.py
Output (figures_export/Request Pruner RHO/): rho_request_pruner_cand_matched_bi40_ss10.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
SRC = "outputs/rho_request_pruner_cand_C2_c2_bi40_ss10_matched.csv"
d = pd.read_csv(SRC)
d = d[d.status == "ok"].copy()
d["cls"] = d.instance.str.extract(r"^[a-z]+(\d)")[0]
N = 726
V = ["base", "reqbase@0.1", "reqplus@0.02", "reqbase@0.2", "reqplus@0.1"]
LAB = {"base": "Baseline\n(no pruner)", "reqbase@0.1": "13 features\nt=0.1\n(~36 % pruned)", "reqplus@0.02": "13 + 4 features\nt=0.02\n(~36 % pruned)",
       "reqbase@0.2": "13 features\nt=0.2\n(~50 % pruned)", "reqplus@0.1": "13 + 4 features\nt=0.1\n(~50 % pruned)"}
COL = {"base": "#8C8C8C", "reqbase@0.1": "#1E2761", "reqplus@0.02": "#E8A33D", "reqbase@0.2": "#5B6BC0", "reqplus@0.1": "#B5731A"}
b = d[d.variant == "base"]
pb = b.groupby("instance").served.mean()
rows = []
for v in V:
    s = d[d.variant == v]
    per = s.groupby("rep").served.sum()
    pv = s.groupby("instance").served.mean()
    r = dict(variant=v, served_mean=per.mean(), rep0=per.iloc[0], rep1=per.iloc[1], rate=100 * per.mean() / N, wall_s=s.wall_s.sum(), speedup=b.wall_s.sum() / s.wall_s.sum(),
             kept_pct=(100 * s.req_kept.sum() / s.req_total.sum()) if s.req_total.sum() > 0 else float("nan"),
             inst_lower=int((pv < pb).sum()) if v != "base" else 0, inst_equal=int((pv == pb).sum()) if v != "base" else 0, inst_higher=int((pv > pb).sum()) if v != "base" else 0)
    for c in ("1", "2"):
        sc, bc = s[s.cls == c], b[b.cls == c]
        r[f"served_c{c}"] = sc.served.sum() / 2
        r[f"speedup_c{c}"] = bc.wall_s.sum() / sc.wall_s.sum()
    rows.append(r)
T = pd.DataFrame(rows).set_index("variant")
T["diff_req"] = T.served_mean - T.loc["base", "served_mean"]
T["diff_pp"] = 100 * T.diff_req / N
T.round(3).to_csv(OUT / "rho_request_pruner_cand_matched_bi40_ss10_table.csv")

fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
a = ax[0]
for i, v in enumerate(V):
    a.bar(i, T.loc[v, "rate"], 0.62, color=COL[v])
    a.scatter([i, i], [100 * T.loc[v, "rep0"] / N, 100 * T.loc[v, "rep1"] / N], color="white", edgecolor="black", s=18, zorder=3)
    a.text(i, max(T.loc[v, "rep0"], T.loc[v, "rep1"]) / N * 100 + 0.12, f"{T.loc[v, 'rate']:.1f} %" + ("" if v == "base" else f"\n({T.loc[v, 'diff_pp']:+.2f} pp,\n{T.loc[v, 'diff_req']:+.1f} req.)"), ha="center", va="bottom", fontsize=8)
a.set_xticks(range(len(V)))
a.set_xticklabels([LAB[v] for v in V], fontsize=7.8)
a.set_ylim(72, 78)
a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
a.set_ylabel("Service rate (mean of 2 reps; dots = single reps)", fontsize=10)
a.set_title("(a) Service rate, 14 instances / 726 requests (y-axis starts at 72 %)", fontsize=10)
bb = ax[1]
for i, v in enumerate(V):
    bb.bar(i, T.loc[v, "speedup"], 0.62, color=COL[v])
    extra = "" if v == "base" else f"\nkept: {T.loc[v, 'kept_pct']:.0f} %\nclass 1: {T.loc[v, 'speedup_c1']:.2f}x\nclass 2: {T.loc[v, 'speedup_c2']:.2f}x"
    bb.text(i, T.loc[v, "speedup"] + 0.004, f"{T.loc[v, 'speedup']:.2f}x{extra}", ha="center", va="bottom", fontsize=8)
bb.axhline(1, color="#444444", lw=0.8, ls=":")
bb.set_xticks(range(len(V)))
bb.set_xticklabels([LAB[v] for v in V], fontsize=7.8)
bb.set_ylim(0.9, 1.4)
bb.set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=10)
bb.set_title("(b) Speed-up (14 instances x 2 reps); kept = share of requests passed on", fontsize=10)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.suptitle("RHO with the Request Pruner without vs. with vehicle-related features at a MATCHED pruning rate, batch_interval=40/step_size=10, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, batch_interval=40, step_size=10, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 Li&Lim test instances, 2 reps, local, runs interleaved, own baseline in the same run.\n"
         "Both pruners: window-specific MLP on the same C=2 data (36 train/8 val/12 test instances), 432 configs x 5 seeds, winners h64/l3/d0.2/pw1.0/lr0.003 (13 features) and h32/l3/d0.2/pw1.0/lr0.03 (17 features).\n"
         "Thresholds matched on the 8 val instances: ~36 % of candidate rows pruned (t=0.1 vs 0.02) and ~50 % (t=0.2 vs 0.1); min_retention_fraction 0.3; active/urgent force-kept; pruned requests are deferred, not rejected. "
         f"+-1 request noise per instance. Source: {SRC}", ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_request_pruner_cand_matched_bi40_ss10.{ext}", dpi=200)
plt.close(fig)

names = {"base": "Baseline (no pruner)", "reqbase@0.1": "13 features, $t{=}0.1$", "reqplus@0.02": "13 + 4 vehicle features, $t{=}0.02$",
         "reqbase@0.2": "13 features, $t{=}0.2$", "reqplus@0.1": "13 + 4 vehicle features, $t{=}0.1$"}
lines = []
for v in V:
    r = T.loc[v]
    lines.append(f"{names[v]} & {r.served_mean:.1f} ({r.rate:.1f}) & " + ("--" if v == "base" else f"{r.diff_req:+.1f} ({r.diff_pp:+.2f})") + f" & {r.speedup:.2f}$\\times$ & " + ("--" if v == "base" else f"{r.kept_pct:.0f}") + " \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the window-specific Request Pruner trained without and with four vehicle-related features at a matched pruning rate (\\texttt{batch\\_interval}=40, \\texttt{step\\_size}=10, $C{=}2$, "
       "cardinality 2, 14 instances, 726 requests, mean of 2 repetitions). Thresholds matched on the 8 validation instances (about 36\\,\\% and 50\\,\\% of the candidate rows pruned). Speed-up = baseline / variant wall-clock; "
       "kept = share of requests passed to trip generation.}\n"
       "\\begin{tabular}{lcccc}\n\\toprule\nVariant & Served (rate \\%) & Diff.\\ to baseline (pp) & Speed-up & Kept (\\%) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_request_pruner_cand_matched_bi40_ss10_table.tex").write_text(tex)
print(T.round(3).to_string())
