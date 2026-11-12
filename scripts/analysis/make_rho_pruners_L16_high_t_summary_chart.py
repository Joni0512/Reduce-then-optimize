"""
2026-11-09: summary over the 4 windows of the high-threshold (0.8 / 0.9) RHO pruner comparison at total limit L=16 (C=2, max_cardinality 2, 14 instances / 726 requests, 2 reps, LRZ serial_std).
Reads the per-window tables written by make_rho_pruners_L16_high_t_chart.py (figures_export/Request Pruner RHO/rho_pruners_L16_high_t_bi<bi>_ss<ss>_table.csv; their source CSVs are
outputs/rho_pair_pruner_runtime_C2_c2_bi<bi>_ss<ss>_L16_cluster_hight.csv). Panel (a): change of the service rate vs. the OWN baseline of that window (pp), panel (b): speed-up vs. own baseline.
Run: ./venv/bin/python3 scripts/analysis/make_rho_pruners_L16_high_t_summary_chart.py
Output: figures_export/Request Pruner RHO/rho_pruners_L16_high_t_summary.{png,pdf}, ..._table.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("figures_export/Request Pruner RHO")
WINS = [("40", "10"), ("100", "25"), ("120", "30"), ("200", "50")]
VARS = ["pair0.8", "pair0.9", "req0.8", "req0.9", "comb0.8_0.8", "comb0.9_0.9"]
LAB = {"pair0.8": "Pair t=0.8", "pair0.9": "Pair t=0.9", "req0.8": "Request t=0.8", "req0.9": "Request t=0.9", "comb0.8_0.8": "Combined 0.8/0.8", "comb0.9_0.9": "Combined 0.9/0.9"}
COL = {"pair0.8": "#E8A33D", "pair0.9": "#B97A12", "req0.8": "#1E2761", "req0.9": "#5B6BC0", "comb0.8_0.8": "#2A9D8F", "comb0.9_0.9": "#17695F"}
T = {w: pd.read_csv(OUT / f"rho_pruners_L16_high_t_bi{w[0]}_ss{w[1]}_table.csv", index_col="variant") for w in WINS}
rows = [dict(window=f"{w[0]}/{w[1]}", variant=v, base_rate=T[w].loc["base", "rate"], diff_pp=T[w].loc[v, "diff_pp"], diff_req=T[w].loc[v, "diff_req"], speedup=T[w].loc[v, "speedup"],
             req_kept=T[w].loc[v, "req_kept"], pair_kept=T[w].loc[v, "pair_kept"]) for w in WINS for v in VARS]
S = pd.DataFrame(rows)
S.round(3).to_csv(OUT / "rho_pruners_L16_high_t_summary_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(15.5, 6))
wl = [f"batch_interval={w[0]}/\nstep_size={w[1]}\n(baseline {T[w].loc['base', 'rate']:.1f} %)" for w in WINS]
bw = 0.13
for k, v in enumerate(VARS):
    xs = [i + (k - 2.5) * bw for i in range(len(WINS))]
    for a, c, fmt in ((ax[0], "diff_pp", "{:+.1f}"), (ax[1], "speedup", "{:.2f}")):
        vals = [S[(S.window == f"{w[0]}/{w[1]}") & (S.variant == v)][c].iloc[0] for w in WINS]
        a.bar(xs, vals, bw, color=COL[v], label=LAB[v])
        for x, y in zip(xs, vals):
            a.text(x, y + (0.08 if c == "diff_pp" and y >= 0 else -0.08 if c == "diff_pp" else 0.004), fmt.format(y), ha="center", va="bottom" if (c == "speedup" or y >= 0) else "top", fontsize=6.3, rotation=90)
for a in ax:
    a.set_xticks(range(len(WINS)))
    a.set_xticklabels(wl, fontsize=8.5)
    a.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].axhline(0, color="#444444", lw=0.8)
ax[0].set_ylim(-5.8, 0.8)
ax[0].set_ylabel("Service rate vs. own baseline (pp)", fontsize=10)
ax[0].set_title("(a) Change of the service rate (negative = worse than baseline)", fontsize=10)
ax[1].axhline(1, color="#444444", lw=0.8, ls=":")
ax[1].set_ylim(0.95, 1.4)
ax[1].set_ylabel("Speed-up = baseline wall-clock / variant wall-clock", fontsize=10)
ax[1].set_title("(b) Speed-up vs. own baseline", fontsize=10)
ax[0].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2)
fig.suptitle("RHO with Pair / Request / Combined Pruner at thresholds 0.8 and 0.9, 4 windows, total limit L=16 per vehicle, C=2, max_cardinality=2", fontsize=11.5)
fig.text(0.5, 0.012, "Config: RHO --mode offline, --max_requests_per_vehicle 16, C=2 on board (unit-demand copy), 14 test instances (726 requests), 2 reps, LRZ serial_std, own baseline per window (runtimes only comparable within a window).\n"
         "Pair Pruner: GNNv1 rgnn_mixed_c2_pw2_v1_final (not window-specific). Request Pruner: window-specific MLP (Rule A checkpoint per window), min_retention_fraction 0.3. Combined = same threshold for both. +-1 request noise per instance.\n"
         "Sources: outputs/rho_pair_pruner_runtime_C2_c2_bi{40_ss10,100_ss25,120_ss30,200_ss50}_L16_cluster_hight.csv; per-window tables in figures_export/Request Pruner RHO/.",
         ha="center", fontsize=6.9, color="#666666")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rho_pruners_L16_high_t_summary.{ext}", dpi=200)
plt.close(fig)
print(S.pivot(index="variant", columns="window", values="diff_pp").reindex(VARS).round(2).to_string())
print(S.pivot(index="variant", columns="window", values="speedup").reindex(VARS).round(2).to_string())
