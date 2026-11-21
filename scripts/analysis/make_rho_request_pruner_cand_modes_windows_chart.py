"""
2026-11-20: chart + table over FOUR windows and THREE pruning modes for the Request Pruner without (13 features) vs. with (13 + 4 vehicle-related candidate-insertion features):
fixed score threshold 0.5, fixed score threshold 0.7, and rank-based 70 % (best 30 % of the candidates per window; threshold 1.01).
RHO offline, C=2 on board (unit-demand copy), max_cardinality 2, no total limit, 14 Li&Lim test instances (726 requests), 2 reps interleaved, each CSV has its OWN baseline (no pruner) from the
same run, the differences are taken against that baseline (baselines differ by up to ~10 requests between runs = RHO noise). min_retention_fraction 0.3, active/urgent requests force-kept,
pruned requests are deferred, not rejected. Both pruners window-specific, same C=2 data, 432 configs x 5 seeds (sweep_request_pruner_cand.py).
Speed-up = baseline wall-clock / variant wall-clock (time.monotonic, sum over 14 instances x 2 reps). kept = requests passed to trip generation / requests seen (sum over windows).
Sources: outputs/rho_request_pruner_cand_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_{t05,t07,t09,p70}.csv
Run: ./venv/bin/python3 scripts/analysis/make_rho_request_pruner_cand_modes_windows_chart.py
Output (figures_export/Request Pruner RHO/): rho_request_pruner_cand_modes_windows.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
WIN = [(40, 10), (100, 25), (120, 30), (200, 50)]
N = 726
MODES = [("t05", "0.5", "threshold 0.5"), ("t07", "0.7", "threshold 0.7"), ("t09", "0.9", "threshold 0.9"), ("p70", "1.01", "rank-based 70 %")]  # 2026-11-21: threshold 0.9 added
# colour = pruner (navy 13 features, amber 13 + 4 features), shade = mode (light -> dark = threshold 0.5, threshold 0.7, rank-based 70 %)
COL = {("base13", 0): "#C5CAE9", ("base13", 1): "#7986CB", ("base13", 2): "#3949AB", ("base13", 3): "#1E2761",
       ("plus", 0): "#FBE7C2", ("plus", 1): "#F1BD6A", ("plus", 2): "#D98E1F", ("plus", 3): "#8A5410"}

rows = []
for bi, ss in WIN:
    for m, (tag, thr, mlab) in enumerate(MODES):
        d = pd.read_csv(f"outputs/rho_request_pruner_cand_C2_c2_bi{bi}_ss{ss}_{tag}.csv")
        d = d[d.status == "ok"].copy()
        b = d[d.variant == "base"]
        bs = b.groupby("rep").served.sum().mean()
        for key, name in (("base13", f"reqbase@{thr}"), ("plus", f"reqplus@{thr}")):
            s = d[d.variant == name]
            per = s.groupby("rep").served.sum()
            rows.append(dict(window=f"{bi}/{ss}", mode=mlab, m=m, pruner=key, baseline=bs, served=per.mean(), diff_req=per.mean() - bs, diff_pp=100 * (per.mean() - bs) / N,
                             rate=100 * per.mean() / N, baseline_rate=100 * bs / N, speedup=b.wall_s.sum() / s.wall_s.sum(), kept_pct=100 * s.req_kept.sum() / s.req_total.sum(),
                             n_runs=len(s)))
T = pd.DataFrame(rows)
T.round(3).to_csv(OUT / "rho_request_pruner_cand_modes_windows_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 6.6))
w = 0.1
order = [(k, m) for m in range(4) for k in ("base13", "plus")]
for k, (key, m) in enumerate(order):
    t = T[(T.pruner == key) & (T.m == m)].set_index("window").loc[[f"{a}/{b}" for a, b in WIN]]
    xs = np.arange(len(WIN)) + (k - 3.5) * w
    ax[0].bar(xs, t.diff_pp, w, color=COL[(key, m)])
    ax[1].bar(xs, t.speedup, w, color=COL[(key, m)])
    for xi, (_, r) in zip(xs, t.iterrows()):
        ax[0].text(xi, r.diff_pp - 0.06, f"{r.diff_req:+.0f}", ha="center", va="top", fontsize=6.2)
        ax[1].text(xi, r.speedup + 0.004, f"{r.speedup:.2f}", ha="center", va="bottom", fontsize=6.3, rotation=90)
for z in ax:
    z.set_xticks(np.arange(len(WIN)))
    z.set_xticklabels([f"batch_interval={a}/\nstep_size={b}" for a, b in WIN], fontsize=9)
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
ax[0].axhline(0, color="#444444", lw=0.9)
ax[0].set_ylim(-3.6, 0.7)
ax[0].set_ylabel("Change in service rate vs. own baseline (pp); number = change in requests", fontsize=9)
ax[0].set_title("(a) Service-rate loss (closer to 0 is better), 14 instances / 726 requests", fontsize=10)
ax[1].axhline(1, color="#444444", lw=0.8, ls=":")
ax[1].set_ylim(0.9, 1.55)
ax[1].set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=9.5)
ax[1].set_title("(b) Speed-up (14 instances x 2 reps)", fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[(key, m)]) for key, m in order]
labels = [f"{'13 features' if key == 'base13' else '13 + 4 vehicle features'}, {MODES[m][2]}" for key, m in order]
ax[0].legend(handles, labels, fontsize=7.4, loc="lower left", ncol=2)
fig.suptitle("RHO with the Request Pruner without vs. with vehicle-related features: fixed thresholds (0.5, 0.7, 0.9) and rank-based 70 %, four windows, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 Li&Lim test instances (726 requests), 2 reps, local, runs interleaved, own baseline per window AND per mode.\n"
         "Pruners: window-specific MLP per window on C=2 data (36 train/8 val/12 test instances), 432 configs x 5 seeds, winner by val; min_retention_fraction 0.3, active/urgent force-kept, pruned = deferred. Rank-based: threshold 1.01 = keep the best 30 % of the candidates per window.\n"
         "Baseline repetitions differ by up to ~10 requests in total (RHO noise, +-1 request per instance), so single differences of a few requests are not significant. Sources: outputs/rho_request_pruner_cand_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_{t05,t07,t09,p70}.csv",
         ha="center", fontsize=6.7, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_request_pruner_cand_modes_windows.{ext}", dpi=200)
plt.close(fig)

lines = []
for bi, ss in WIN:
    for m, (tag, thr, mlab) in enumerate(MODES):
        a = T[(T.window == f"{bi}/{ss}") & (T.m == m) & (T.pruner == "base13")].iloc[0]
        p = T[(T.window == f"{bi}/{ss}") & (T.m == m) & (T.pruner == "plus")].iloc[0]
        lines.append(f"{bi}/{ss} & {mlab} & {a.baseline:.1f} & {a.diff_req:+.1f} ({a.diff_pp:+.2f}) & {a.speedup:.2f}$\\times$ & {p.diff_req:+.1f} ({p.diff_pp:+.2f}) & {p.speedup:.2f}$\\times$ \\\\")
    lines.append("\\midrule")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the window-specific Request Pruner trained with 13 features or with 13 plus four vehicle-related features, at fixed score thresholds (0.5, 0.7, 0.9) and rank-based 70\\,\\% pruning "
       "($C{=}2$, cardinality 2, 14 instances, 726 requests, mean of 2 repetitions). Window = \\texttt{batch\\_interval}/\\texttt{step\\_size}. Changes are relative to the baseline without pruner of the same run "
       "(requests, percentage points in brackets). Speed-up = baseline / variant wall-clock.}\n"
       "\\begin{tabular}{llcccccc}\n\\toprule\nWindow & Pruning & Baseline (req.) & \\multicolumn{2}{c}{13 features} & \\multicolumn{2}{c}{13 + 4 features} \\\\\n"
       " & & & Change & Speed-up & Change & Speed-up \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
tex = tex.replace("\\begin{tabular}{llcccccc}", "\\begin{tabular}{llccccc}")
(OUT / "rho_request_pruner_cand_modes_windows_table.tex").write_text(tex)
print(T.round(2).to_string())
print(T.groupby(["mode", "pruner"])[["diff_req", "speedup"]].mean().round(2))
