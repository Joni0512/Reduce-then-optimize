"""
2026-11-21: chart + table: Pair Pruner, Request Pruner and BOTH together, with the Request Pruner trained without (13 features) vs. with (13 + 4 vehicle-related features), four windows.
RHO offline, C=2 on board (unit-demand copy), max_cardinality 2, no total limit, 14 Li&Lim test instances (726 requests), 2 reps interleaved, sequential local runs.
Pair Pruner = existing GLOBAL GNN rgnn_mixed_c2_pw2_v1_final (no vehicle features, not window-specific), threshold 0.5. Request Pruner = window-specific, fixed threshold 0.5, min_retention_fraction 0.3,
active/urgent force-kept, pruned = deferred. The pair pruner scores ALL requests of the window (the 13 + 4 request pruner is deferred until the single-trip costs exist), see rho_pair_request_cand_run.py.
Two runs per window, each with its OWN baseline (no pruner) in the same run; every difference is taken against that baseline (baselines differ by up to ~10 requests between runs = RHO noise):
  run A (outputs/rho_pair_request_cand_C2_c2_bi<BI>_ss<SS>_p05r05.csv):        base | pair0.5 | reqplus0.5 | comb0.5_0.5 (Pair + Request 13 + 4)
  run B (outputs/rho_pair_request_cand_C2_c2_bi<BI>_ss<SS>_p05r05_base13.csv): base | reqbase0.5 | combbase0.5_0.5 (Pair + Request 13)
Speed-up = baseline wall-clock / variant wall-clock (time.monotonic, sum over 14 instances x 2 reps).
Run: ./venv/bin/python3 scripts/analysis/make_rho_pair_request_cand_chart.py
Output (figures_export/Request Pruner RHO/): rho_pair_request_cand_windows.{png,pdf}, ..._table.csv, ..._table.tex
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
OUT.mkdir(parents=True, exist_ok=True)
WIN = [(40, 10), (100, 25), (120, 30), (200, 50)]
N = 726
# (key, run, variant name in the CSV, label)
VARS = [("pair", "A", "pair0.5", "Pair Pruner only"),
        ("req13", "B", "reqbase0.5", "Request Pruner only, 13 features"),
        ("req17", "A", "reqplus0.5", "Request Pruner only, 13 + 4 features"),
        ("comb13", "B", "combbase0.5_0.5", "Pair + Request Pruner, 13 features"),
        ("comb17", "A", "comb0.5_0.5", "Pair + Request Pruner, 13 + 4 features")]
COL = {"pair": "#8C8C8C", "req13": "#5B6BC0", "req17": "#E8A33D", "comb13": "#1E2761", "comb17": "#A86A12"}

rows = []
for bi, ss in WIN:
    runs = {"A": pd.read_csv(f"outputs/rho_pair_request_cand_C2_c2_bi{bi}_ss{ss}_p05r05.csv"), "B": pd.read_csv(f"outputs/rho_pair_request_cand_C2_c2_bi{bi}_ss{ss}_p05r05_base13.csv")}
    for k in runs:
        runs[k] = runs[k][runs[k].status == "ok"]
    for key, run, name, lab in VARS:
        d = runs[run]
        b, s = d[d.variant == "base"], d[d.variant == name]
        bs, per = b.groupby("rep").served.sum().mean(), s.groupby("rep").served.sum().mean()
        rows.append(dict(window=f"{bi}/{ss}", key=key, label=lab, run=run, baseline=bs, served=per, diff_req=per - bs, diff_pp=100 * (per - bs) / N, speedup=b.wall_s.sum() / s.wall_s.sum(),
                         req_kept_pct=(100 * s.req_kept.sum() / s.req_total.sum()) if s.req_total.sum() > 0 else np.nan,
                         pairs_kept_pct=(100 * s.pairs_kept.sum() / s.pairs_total.sum()) if s.pairs_total.sum() > 0 else np.nan, n_runs=len(s)))
T = pd.DataFrame(rows)
T.round(3).to_csv(OUT / "rho_pair_request_cand_windows_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
w = 0.16
for k, (key, run, name, lab) in enumerate(VARS):
    t = T[T.key == key].set_index("window").loc[[f"{a}/{b}" for a, b in WIN]]
    xs = np.arange(len(WIN)) + (k - 2) * w
    ax[0].bar(xs, t.diff_pp, w, color=COL[key], label=lab)
    ax[1].bar(xs, t.speedup, w, color=COL[key], label=lab)
    for xi, (_, r) in zip(xs, t.iterrows()):
        ax[0].text(xi, (r.diff_pp + 0.06) if r.diff_pp >= 0 else (r.diff_pp - 0.06), f"{r.diff_req:+.0f}", ha="center", va="bottom" if r.diff_pp >= 0 else "top", fontsize=7)
        ax[1].text(xi, r.speedup + 0.008, f"{r.speedup:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
for z in ax:
    z.set_xticks(np.arange(len(WIN)))
    z.set_xticklabels([f"batch_interval={a}/\nstep_size={b}" for a, b in WIN], fontsize=9)
    z.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        z.spines[sp].set_visible(False)
ax[0].axhline(0, color="#444444", lw=0.9)
ax[0].set_ylim(-4.4, 1.8)  # 2026-11-21: more room at the bottom so the legend does not cover the bars
ax[0].set_ylabel("Change in service rate vs. own baseline (pp); number = change in requests", fontsize=9)
ax[0].set_title("(a) Service-rate change (closer to 0 / above is better), 14 instances / 726 requests", fontsize=10)
ax[0].legend(fontsize=7.6, loc="lower left", ncol=2)
ax[1].axhline(1, color="#444444", lw=0.8, ls=":")
ax[1].set_ylim(0.7, 1.7)
ax[1].set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=9.5)
ax[1].set_title("(b) Speed-up (14 instances x 2 reps)", fontsize=10)
fig.suptitle("RHO with Pair Pruner, Request Pruner and both, Request Pruner without (13) vs. with vehicle features (13 + 4), threshold 0.5, four windows, C=2, max_cardinality=2", fontsize=11)
fig.text(0.5, 0.012, "Config: RHO --mode offline, max_cardinality 2, C=2 on board (unit-demand copy), no total limit, 14 Li&Lim test instances (726 requests), 2 reps, local, sequential, interleaved; own baseline in each of the two runs per window.\n"
         "Pair Pruner: global GNN rgnn_mixed_c2_pw2_v1_final, threshold 0.5, no vehicle features. Request Pruner: window-specific MLP (432 configs x 5 seeds, winner by val), fixed threshold 0.5, min_retention_fraction 0.3, active/urgent force-kept, pruned = deferred.\n"
         "The pair pruner scores all requests of the window (the 13 + 4 request pruner filters only after the single-trip costs). Baselines differ by up to ~10 requests between runs (RHO noise). Sources: outputs/rho_pair_request_cand_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_p05r05{,_base13}.csv",
         ha="center", fontsize=6.7, color="#666666")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_pair_request_cand_windows.{ext}", dpi=200)
plt.close(fig)

lines = []
for bi, ss in WIN:
    for key, run, name, lab in VARS:
        r = T[(T.window == f"{bi}/{ss}") & (T.key == key)].iloc[0]
        lines.append(f"{bi}/{ss} & {lab} & {r.baseline:.1f} & {r.diff_req:+.1f} ({r.diff_pp:+.2f}) & {r.speedup:.2f}$\\times$ \\\\")
    lines.append("\\midrule")
tex = ("\\begin{table}[h]\n\\centering\n\\small\n"
       "\\caption{RHO with the Pair Pruner (global GNN, threshold 0.5), the window-specific Request Pruner (threshold 0.5, 13 or 13 plus four vehicle-related features) and both together "
       "($C{=}2$, cardinality 2, 14 instances, 726 requests, mean of 2 repetitions). Window = \\texttt{batch\\_interval}/\\texttt{step\\_size}. Changes are relative to the baseline without pruner of the same run "
       "(requests, percentage points in brackets). Speed-up = baseline / variant wall-clock.}\n"
       "\\begin{tabular}{llccc}\n\\toprule\nWindow & Variant & Baseline (req.) & Change & Speed-up \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(OUT / "rho_pair_request_cand_windows_table.tex").write_text(tex)
print(T[["window", "key", "baseline", "diff_req", "diff_pp", "speedup", "req_kept_pct", "pairs_kept_pct", "n_runs"]].round(2).to_string(index=False))
print(T.groupby("key")[["diff_req", "speedup"]].mean().round(2))
