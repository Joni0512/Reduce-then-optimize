"""
2026-11-20: chart + table over FOUR windows for the Request Pruner without (13 features) vs. with (13 + 4 vehicle-related candidate-insertion features) at 70 % rank-based pruning.
RHO offline, C=2 on board (unit-demand copy), max_cardinality 2, no total limit, 14 Li&Lim test instances (726 requests), 2 reps interleaved, own baseline in the same run per window.
Pruner variants: reqbase@1.01 / reqplus@1.01 = threshold > 1 => rank-based, the best 30 % of the non-force-kept candidates per window are kept (min_retention_fraction 0.3 is binding),
active/urgent requests force-kept; pruned requests are deferred, not rejected. Both pruners window-specific, same C=2 data, 432 configs x 5 seeds (sweep_request_pruner_cand.py).
Speed-up = baseline wall-clock / variant wall-clock (time.monotonic, sum over 14 instances x 2 reps). kept = requests passed to trip generation / requests seen (sum over windows).
Sources: outputs/rho_request_pruner_cand_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_p70.csv
Run: ./venv/bin/python3 scripts/analysis/make_rho_request_pruner_cand_windows_chart.py
Output (figures_export/Request Pruner RHO/): rho_request_pruner_cand_p70_windows.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
WIN = [(40, 10), (100, 25), (120, 30), (200, 50)]
N = 726
V = ["base", "reqbase@1.01", "reqplus@1.01"]
LAB = {"base": "Baseline (no pruner)", "reqbase@1.01": "13 features, 70 % pruned", "reqplus@1.01": "13 + 4 vehicle features, 70 % pruned"}
COL = {"base": "#8C8C8C", "reqbase@1.01": "#1E2761", "reqplus@1.01": "#E8A33D"}

rows = []
for bi, ss in WIN:
    src = f"outputs/rho_request_pruner_cand_C2_c2_bi{bi}_ss{ss}_p70.csv"
    d = pd.read_csv(src)
    d = d[d.status == "ok"].copy()
    d["cls"] = d.instance.str.extract(r"^[a-z]+(\d)")[0]
    b = d[d.variant == "base"]
    pb = b.groupby("instance").served.mean()
    for v in V:
        s = d[d.variant == v]
        per = s.groupby("rep").served.sum()
        pv = s.groupby("instance").served.mean()
        r = dict(window=f"{bi}/{ss}", bi=bi, ss=ss, variant=v, served=per.mean(), rep0=per.iloc[0], rep1=per.iloc[1], rate=100 * per.mean() / N,
                 wall_s=s.wall_s.sum(), speedup=b.wall_s.sum() / s.wall_s.sum(),
                 kept_pct=(100 * s.req_kept.sum() / s.req_total.sum()) if s.req_total.sum() > 0 else float("nan"),
                 inst_lower=int((pv < pb).sum()) if v != "base" else 0, inst_equal=int((pv == pb).sum()) if v != "base" else 0, inst_higher=int((pv > pb).sum()) if v != "base" else 0)
        for c in ("1", "2"):
            sc, bc = s[s.cls == c], b[b.cls == c]
            r[f"served_c{c}"] = sc.served.sum() / 2
            r[f"speedup_c{c}"] = bc.wall_s.sum() / sc.wall_s.sum()
        rows.append(r)
T = pd.DataFrame(rows)
T["diff_req"] = T.served - T.groupby("window").served.transform(lambda x: x.iloc[0])
T["diff_pp"] = 100 * T.diff_req / N
T.round(3).to_csv(OUT / "rho_request_pruner_cand_p70_windows_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 6.4))
w = 0.26
x = np.arange(len(WIN))
for k, v in enumerate(V):
    t = T[T.variant == v].set_index("window").loc[[f"{a}/{b}" for a, b in WIN]]
    ax[0].bar(x + (k - 1) * w, t.rate, w, color=COL[v], label=LAB[v])
    ax[0].scatter(np.r_[x + (k - 1) * w, x + (k - 1) * w], np.r_[100 * t.rep0 / N, 100 * t.rep1 / N], color="white", edgecolor="black", s=14, zorder=3)
    for xi, (_, r) in zip(x + (k - 1) * w, t.iterrows()):
        ax[0].text(xi, max(r.rep0, r.rep1) / N * 100 + 0.1, f"{r.rate:.1f}" + ("" if v == "base" else f"\n{r.diff_req:+.1f}"), ha="center", va="bottom", fontsize=7.5)
    ax[1].bar(x + (k - 1) * w, t.speedup, w, color=COL[v], label=LAB[v])
    for xi, (_, r) in zip(x + (k - 1) * w, t.iterrows()):
        ax[1].text(xi, r.speedup + 0.01, f"{r.speedup:.2f}x", ha="center", va="bottom", fontsize=7.5)
for z in ax:
    z.set_xticks(x)
    z.set_xticklabels([f"batch_interval={a}/\nstep_size={b}" for a, b in WIN], fontsize=8.5)
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
ax[0].set_ylim(62, 80)
ax[0].set_ylabel("Service rate in % (mean of 2 reps; dots = single reps; second number = diff. in requests)", fontsize=8.5)
ax[0].set_title("(a) Service rate, 14 instances / 726 requests (y-axis starts at 62 %)", fontsize=10)
ax[0].legend(fontsize=8, loc="upper right")
ax[1].axhline(1, color="#444444", lw=0.8, ls=":")
ax[1].set_ylim(0.9, 1.6)
ax[1].set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=9.5)
ax[1].set_title("(b) Speed-up (14 instances x 2 reps)", fontsize=10)
fig.suptitle("RHO with the Request Pruner without vs. with vehicle-related features, 70 % rank-based pruning, four windows, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 Li&Lim test instances (726 requests), 2 reps, local, runs interleaved, own baseline per window.\n"
         "Pruners: window-specific MLP per window on C=2 data (36 train/8 val/12 test instances), 432 configs x 5 seeds, winner by val; threshold 1.01 = rank-based (keep best 30 % of non-force-kept candidates per window), min_retention_fraction 0.3, active/urgent force-kept.\n"
         "+-1 request noise per instance (rep spread of the baseline up to 6 requests in total). Sources: outputs/rho_request_pruner_cand_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_p70.csv", ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_request_pruner_cand_p70_windows.{ext}", dpi=200)
plt.close(fig)

names = {"base": "Baseline (no pruner)", "reqbase@1.01": "13 features", "reqplus@1.01": "13 + 4 vehicle features"}
lines = []
for bi, ss in WIN:
    for v in V:
        r = T[(T.window == f"{bi}/{ss}") & (T.variant == v)].iloc[0]
        lines.append(f"{bi}/{ss} & {names[v]} & {r.served:.1f} ({r.rate:.1f}) & " + ("--" if v == "base" else f"{r.diff_req:+.1f} ({r.diff_pp:+.2f})") + f" & {r.speedup:.2f}$\\times$ & " + ("--" if v == "base" else f"{r.kept_pct:.0f}") + " \\\\")
    lines.append("\\midrule")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the window-specific Request Pruner at 70\\,\\% rank-based pruning, trained without and with four vehicle-related features ($C{=}2$, cardinality 2, 14 instances, 726 requests, mean of 2 repetitions). "
       "Window = \\texttt{batch\\_interval}/\\texttt{step\\_size}. Speed-up = baseline / variant wall-clock; kept = share of requests passed to trip generation.}\n"
       "\\begin{tabular}{llcccc}\n\\toprule\nWindow & Variant & Served (rate \\%) & Diff.\\ to baseline (pp) & Speed-up & Kept (\\%) \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_request_pruner_cand_p70_windows_table.tex").write_text(tex)
print(T.round(3).to_string())
