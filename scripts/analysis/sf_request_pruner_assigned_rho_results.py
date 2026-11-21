"""
2026-11-21: SF request pruner with the "assigned" label - RHO test results on 4 window configs (user 2026-11-20:
"probiere dieses Label bei SF mehrere Time Windows"; thresholds 0.6/0.7/0.8, min_retention_fraction 0.0).
Runs: submit_sf_lodes_rho_uniformroad_reqpruner_assigned_allwindows.sbatch, cluster array sf_ur_rpa, tasks 0-44 and 60-74
(1800/900 NOT run: its pruner has val AUC 0.54 and would prune 98 % of requests at 0.6 offline).
Setup: SF test instance seed 42, start seeds 42-46 (uniform road-node starts), 150 vehicles, max_cardinality=2,
time rule off, keep_active False, no depot return, rtv_timeout 1800 s (3600/1800: 21600 s), geographic pruner,
one pruner per window (outputs/request_pruner_mlp_sf_assigned_bi{BI}_ss{SS}/, train instance seed 1, val seed 2).
Input: outputs/reqpruner_assigned_sf_results.csv (request pruner) + outputs/pairpruner_sf_results.csv (BASE rows = plain RHO,
same start seeds) - both written on the cluster by the inline export in chat from final/results.json.
Output: figures_export/SF/sf_request_pruner_assigned_rho_results.{png,pdf} + _table.csv
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export/SF/sf_request_pruner_assigned_rho_results"
rp = list(csv.DictReader(open(ROOT / "outputs/reqpruner_assigned_sf_results.csv")))
base = [r for r in csv.DictReader(open(ROOT / "outputs/pairpruner_sf_results.csv")) if r["label"] == "BASE"]
CONFIGS = ["bi300_ss150", "bi600_ss300", "bi900_ss450", "bi3600_ss1800_rtv21600"]
THRS = ["0.6", "0.7", "0.8"]


def nice(c):
    bi, ss = c[2:].split("_ss")
    rtv = " (rtv 21600 s)" if "rtv21600" in ss else ""
    return f"batch_interval={bi}/\nstep_size={ss.replace('_rtv21600', '')}{rtv}"


table = []
for c in CONFIGS:
    b = [r for r in base if r["config"] == c]
    b_sr, b_t = st.mean(float(r["service_rate"]) for r in b), st.mean(float(r["total_time_s"]) for r in b)
    table.append({"config": c, "threshold": "baseline", "n": len(b), "sr_mean": round(b_sr, 2), "sr_min": "", "sr_max": "",
                  "delta_pp": 0.0, "time_mean_s": round(b_t, 1), "time_pct_of_baseline": 100.0})
    for t in THRS:
        s = [r for r in rp if r["config"] == c and r["threshold"] == t]
        sr = [float(r["service_rate"]) for r in s]; tt = [float(r["total_time_s"]) for r in s]
        table.append({"config": c, "threshold": t, "n": len(s), "sr_mean": round(st.mean(sr), 2), "sr_min": min(sr),
                      "sr_max": max(sr), "delta_pp": round(st.mean(sr) - b_sr, 2), "time_mean_s": round(st.mean(tt), 1),
                      "time_pct_of_baseline": round(100 * st.mean(tt) / b_t, 1)})
with open(OUT.parent / (OUT.name + "_table.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

COL = {"0.6": "#9ec5f8", "0.7": "#2a78d6", "0.8": "#123f7a"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(CONFIGS)); wb = 0.26
for j, t in enumerate(THRS):
    rows = [next(r for r in table if r["config"] == c and r["threshold"] == t) for c in CONFIGS]
    for a, key, fmt in ((ax[0], "delta_pp", "{:+.1f}"), (ax[1], "time_pct_of_baseline", "{:.0f}%")):
        v = [r[key] for r in rows]
        a.bar(x + (j - 1) * wb, v, wb, color=COL[t], edgecolor="white", linewidth=1.5, label=f"threshold {t}")
        for xi, vi in zip(x + (j - 1) * wb, v):
            a.text(xi, vi + (0.4 if (key == "delta_pp" and vi >= 0) else -0.4 if key == "delta_pp" else 1.5), fmt.format(vi),
                   ha="center", va="bottom" if (key != "delta_pp" or vi >= 0) else "top", fontsize=8)
ax[0].axhline(0, color="#444", lw=1); ax[1].axhline(100, color="#444", lw=1, ls="--")
ax[0].set_ylabel("Δ service rate vs. RHO baseline (pp)"); ax[1].set_ylabel("Runtime (% of RHO baseline)")
for a in ax:
    a.set_xticks(x); a.set_xticklabels([nice(c) for c in CONFIGS], fontsize=8)
    a.grid(axis="y", color="#e5e5e5"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("SF request pruner, label 'assigned' (one pruner per window), RHO on test instance seed 42, mean over start seeds 42-46\n"
             "max_cardinality=2, 150 vehicles, keep_active False, time rule off, min_retention_fraction 0.0, rtv_timeout 1800 s unless noted",
             fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200)
for r in table: print(r)
