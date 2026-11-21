"""
2026-11-20: SF request pruner RHO test vs. time rule vs. RHO baseline (no pruning), batch_interval=900/step_size=450.
Test instance seed 42, start seeds 42-46 (uniform road-node starts), max_cardinality=2, 150 vehicles, keep_active False,
no depot return, rtv_timeout 1800.
Variants:
  - baseline: no pruning, no time rule (outputs/new_tests/sf_lodes_nodepot/bi900_ss450_mc2_v150_notimerule_noreturn_uniformroad_s*)
  - time rule: --use_request_time_rule_filter True (cluster jobs 5743073_5-9 / 5743600-03)
  - request pruner: label pickup_before_next_resolve, MLP h64/l1/d0.2/pw1.0/lr0.03, threshold 0.5,
    min_retention_fraction 0.0, geographic True (cluster jobs 5743964_0-4)
    (outputs/new_tests/sf_lodes_nodepot/bi900_ss450_mc2_v150_noreturn_uniformroad_{timerule,reqpruner_pickup_t0.5_minret0}_s*)
Values = serviced/total_requests and stats.total_time from each run's final/results.json, read on the LRZ cluster
(user paste 2026-11-20); hard-coded here because the run dirs are not synced locally.
Output: figures_export/SF/sf_request_pruner_rho_results_bi900_ss450.{png,pdf,csv}
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export/SF/sf_request_pruner_rho_results_bi900_ss450"
SEEDS = [42, 43, 44, 45, 46]
DATA = {  # variant -> (service rate % per start seed, runtime s per start seed)
    "RHO baseline\n(no pruning)": ([89.4, 90.3, 91.5, 89.6, 90.5], [478, 445, 454, 451, 445]),
    "Time rule": ([74.5, 73.1, 76.0, 74.7, 73.3], [147, 154, 157, 142, 140]),
    "Request pruner\n(pickup label, thr 0.5)": ([67.9, 68.1, 69.6, 70.4, 67.3], [149, 126, 132, 132, 156]),
}
COLORS = ["#8a8f98", "#2a78d6", "#d9822b"]

rows = []
base_sr = np.mean(DATA["RHO baseline\n(no pruning)"][0]); base_rt = np.mean(DATA["RHO baseline\n(no pruning)"][1])
for name, (sr, rt) in DATA.items():
    rows.append({"variant": name.replace("\n", " "), "service_rate_mean_pct": round(np.mean(sr), 2),
                 "service_rate_min": min(sr), "service_rate_max": max(sr),
                 "delta_pp_vs_baseline": round(np.mean(sr) - base_sr, 2), "runtime_mean_s": round(np.mean(rt), 1),
                 "runtime_share_of_baseline_pct": round(100 * np.mean(rt) / base_rt, 1),
                 **{f"sr_s{s}": v for s, v in zip(SEEDS, sr)}, **{f"rt_s{s}": v for s, v in zip(SEEDS, rt)}})
with open(OUT.with_suffix(".csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, idx, ylab in ((axes[0], 0, "Service rate (%)"), (axes[1], 1, "Runtime (s)")):
    for i, (name, vals) in enumerate(DATA.items()):
        v = vals[idx]
        ax.bar(i, np.mean(v), width=0.6, color=COLORS[i], edgecolor="white", linewidth=2)
        ax.scatter(np.full(5, i) + np.linspace(-0.15, 0.15, 5), v, s=22, color="#222", zorder=3)
        lab = f"{np.mean(v):.2f} %" if idx == 0 else f"{np.mean(v):.0f} s"
        ax.text(i, np.mean(v) * 0.5, lab, ha="center", va="center", fontsize=11, color="white", fontweight="bold")  # 2026-11-20: inside bar, no overlap with seed dots
    ax.set_xticks(range(3)); ax.set_xticklabels(DATA.keys(), fontsize=9); ax.set_ylabel(ylab)
    ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylim(0, 100)
fig.suptitle("SF RHO, batch_interval=900/step_size=450, max_cardinality=2, 150 vehicles, keep_active False, rtv_timeout 1800 s\n"
             "test instance seed 42, start seeds 42-46 (dots), bars = mean; request pruner min_retention_fraction 0.0",
             fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200)
for r in rows: print(r)
