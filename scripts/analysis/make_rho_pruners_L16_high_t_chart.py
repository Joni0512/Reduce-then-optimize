"""
2026-11-09: RHO pruner comparison with HIGH thresholds 0.8 and 0.9 (Pair, Request, Combined) vs. own baseline, one window per call (--bi/--ss), total limit L=16 per vehicle,
C=2 on board (unit-demand manifest copy), max_cardinality 2, 14 Li&Lim test instances (726 requests), 2 repetitions, runs interleaved per instance, cluster (LRZ serial_std).
Input: outputs/rho_pair_pruner_runtime_C2_c2_bi<bi>_ss<ss>_L16_cluster_hight.csv (from submit_rho_pruners_windows_L16_high_t.sbatch -> rho_pair_pruner_runtime.py).

Variants: base | pair<t> | req<t> | comb<t>_<t> (same threshold for both). Pair Pruner = GNNv1 rgnn_mixed_c2_pw2_v1_final (NOT window-specific); Request Pruner = window-specific MLP, checkpoint
chosen by Rule A (outputs/request_pruner_threshold_rule_a_bi<bi>_ss<ss>.json), min_retention_fraction 0.3. RHO is not bit-deterministic (+-1 request per instance = noise); dots = the 2 reps.
Speed-up = baseline wall-clock / variant wall-clock (sum over 14 instances and 2 reps, subprocess wall-clock incl. pruner inference); runtimes only comparable within this job.

Run: ./venv/bin/python3 scripts/analysis/make_rho_pruners_L16_high_t_chart.py --bi 40 --ss 10
Output (figures_export/Request Pruner RHO/): rho_pruners_L16_high_t_bi<bi>_ss<ss>.{png,pdf}, ..._table.csv, ..._table.tex
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bi", default="40")
ap.add_argument("--ss", default="10")
A = ap.parse_args()
BI, SS = A.bi, A.ss
OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
SRC = f"outputs/rho_pair_pruner_runtime_C2_c2_bi{BI}_ss{SS}_L16_cluster_hight.csv"
d = pd.read_csv(SRC)
d = d[d.status == "ok"]
N_REQ = 726
ORDER = ["base", "pair0.8", "pair0.9", "req0.8", "req0.9", "comb0.8_0.8", "comb0.9_0.9"]
LAB = {"base": "Baseline\n(no pruner)", "pair0.8": "Pair\nt=0.8", "pair0.9": "Pair\nt=0.9", "req0.8": "Request\nt=0.8", "req0.9": "Request\nt=0.9",
       "comb0.8_0.8": "Combined\nt=0.8/0.8", "comb0.9_0.9": "Combined\nt=0.9/0.9"}
COL = {"base": "#8C8C8C", "pair": "#E8A33D", "req": "#1E2761", "comb": "#2A9D8F"}
col = lambda v: COL["base"] if v == "base" else COL["pair"] if v.startswith("pair") else COL["req"] if v.startswith("req") else COL["comb"]
ck = json.load(open(f"outputs/request_pruner_threshold_rule_a_bi{BI}_ss{SS}.json"))["checkpoint"].split("/")[-2].replace("request_pruner_mlp_", "")

rows = []
for v in ORDER:
    s = d[d.variant == v]
    per_rep = s.groupby("rep").served.sum()
    rows.append(dict(variant=v, served_mean=per_rep.mean(), served_rep0=per_rep.iloc[0], served_rep1=per_rep.iloc[1], rate=100 * per_rep.mean() / N_REQ,
                     wall_s=s.wall_s.sum(), rtv_s=s.rtv_s.sum(),
                     pair_kept=100 * s.pairs_kept.sum() / s.pairs_total.sum() if s.pairs_total.sum() > 0 else float("nan"),
                     req_kept=100 * s.req_kept.sum() / s.req_total.sum() if s.req_total.sum() > 0 else float("nan")))
t = pd.DataFrame(rows).set_index("variant")
t["speedup"] = t.loc["base", "wall_s"] / t.wall_s
t["diff_req"] = t.served_mean - t.loc["base", "served_mean"]
t["diff_pp"] = 100 * t.diff_req / N_REQ
p = d.groupby(["instance", "variant"]).served.mean().unstack()
t["inst_lower"] = [(p[v] < p["base"]).sum() if v != "base" else 0 for v in ORDER]
t["inst_equal"] = [(p[v] == p["base"]).sum() if v != "base" else 0 for v in ORDER]
t["inst_higher"] = [(p[v] > p["base"]).sum() if v != "base" else 0 for v in ORDER]
t.round(3).to_csv(OUT / f"rho_pruners_L16_high_t_bi{BI}_ss{SS}_table.csv")

fig, ax = plt.subplots(1, 2, figsize=(15, 6.3))
x = range(len(ORDER))
a = ax[0]
for i, v in enumerate(ORDER):
    a.bar(i, t.loc[v, "rate"], 0.7, color=col(v))
    a.scatter([i, i], [100 * t.loc[v, "served_rep0"] / N_REQ, 100 * t.loc[v, "served_rep1"] / N_REQ], color="white", edgecolor="black", s=18, zorder=3)
    txt = f"{t.loc[v, 'rate']:.1f} %" + ("" if v == "base" else f"\n({t.loc[v, 'diff_pp']:+.1f} pp)")
    a.text(i, 100 * t.loc[v, "served_rep1"] / N_REQ + 0.25 if False else max(100 * t.loc[v, "served_rep0"] / N_REQ, 100 * t.loc[v, "served_rep1"] / N_REQ) + 0.25, txt, ha="center", va="bottom", fontsize=8)
a.set_xticks(list(x))
a.set_xticklabels([LAB[v] for v in ORDER], fontsize=8)
a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
a.set_ylim(66, 80)
a.set_ylabel("Service rate (mean of 2 reps; dots = single reps)", fontsize=10)
a.set_title("(a) Service rate, 14 instances / 726 requests (y-axis starts at 66 %)", fontsize=10)
b = ax[1]
for i, v in enumerate(ORDER):
    b.bar(i, t.loc[v, "speedup"], 0.7, color=col(v))
    kept = []
    if v != "base":
        if v.startswith(("pair", "comb")):
            kept.append(f"pairs {t.loc[v, 'pair_kept']:.0f} %")
        if v.startswith(("req", "comb")):
            kept.append(f"req. {t.loc[v, 'req_kept']:.0f} %")
    b.text(i, t.loc[v, "speedup"] + 0.005, f"{t.loc[v, 'speedup']:.2f}x" + ("\nkept:\n" + "\n".join(kept) if kept else ""), ha="center", va="bottom", fontsize=7.6)  # 2026-11-09: one item per line, the single-line version overlapped the neighbouring bar
b.set_xticks(list(x))
b.set_xticklabels([LAB[v] for v in ORDER], fontsize=8)
b.axhline(1, color="#444444", lw=0.8, ls=":")
b.set_ylim(0.9, 1.3)
b.set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=10)
b.set_title('(b) Speed-up (14 instances x 2 reps); kept = share of pairs / requests kept', fontsize=10)
for z in ax:
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
fig.suptitle(f"RHO with Pair / Request / Combined Pruner at high thresholds, batch_interval={BI}/step_size={SS}, total limit L=16 per vehicle, C=2, max_cardinality=2", fontsize=11.5)
fig.text(0.5, 0.012,
         f"Config: RHO --mode offline, batch_interval={BI}, step_size={SS}, --max_requests_per_vehicle 16, C=2 on board (unit-demand copy), max_cardinality 2, 14 test instances, 2 reps, LRZ serial_std (one run at a time).\n"
         f"Pair Pruner: GNNv1 rgnn_mixed_c2_pw2_v1_final (not window-specific). Request Pruner: window-specific MLP {ck} (hidden 64, 3 layers, dropout 0, pos_weight 3, lr 0.03; Rule A checkpoint), "
         "min_retention_fraction 0.3.\n"
         f"Combined = same threshold for both pruners. +-1 request noise per instance. Source: {SRC}",
         ha="center", fontsize=7, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_pruners_L16_high_t_bi{BI}_ss{SS}.{ext}", dpi=200)
plt.close(fig)

lines = []
for v in ORDER:
    r = t.loc[v]
    nm = LAB[v].replace("\n", " ").replace("t=", "$t{=}$").replace("/", "/")
    lines.append(f"{nm} & {r.served_mean:.1f} ({r.rate:.1f}) & " + ("--" if v == "base" else f"{r.diff_req:+.1f} ({r.diff_pp:+.1f})") + f" & {r.speedup:.2f}$\\times$ & "
                 + ("--" if pd.isna(r.pair_kept) else f"{r.pair_kept:.0f}") + " & " + ("--" if pd.isna(r.req_kept) else f"{r.req_kept:.0f}") + " \\\\")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       f"\\caption{{RHO with Pair, Request and Combined Pruner at high thresholds, \\texttt{{batch\\_interval}}={BI}, \\texttt{{step\\_size}}={SS}, total limit $L{{=}}16$, $C{{=}}2$, cardinality 2, 14 instances (726 requests), "
       "mean of 2 repetitions. Speed-up = baseline wall-clock / variant wall-clock. Kept = share of candidate pairs / requests kept by the pruner.}\n"
       "\\begin{tabular}{lcccrr}\n\\toprule\nVariant & Served (rate \\%) & Diff.\\ to baseline (pp) & Speed-up & Pairs kept (\\%) & Requests kept (\\%) \\\\\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / f"rho_pruners_L16_high_t_bi{BI}_ss{SS}_table.tex").write_text(tex)
print(t.round(2).to_string())
