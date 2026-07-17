"""
2026-07-16: COAML mode="optimal" ("OPT (Solution Projection with Rolling Horizon)" in
this report's naming - see chat: user wants this label instead of "OPT"),
threshold 0.5, averaged over the 12 test instances - direct counterpart to
plot_rho_pruner_summary.py (RHO ohne Learning). Reuses the t=0.5 sweep
already run earlier this session (outputs/opt_single_instance/bi200_ss100/
<instance>/{baseline,request_pruner,pair_pruner,both_pruners}/results.json)
rather than re-solving 48 instances that would reproduce the same numbers.

4 of 48 runs crashed (lc207 baseline+pair, lc208 request+both - see chat
history for root causes: a pre-existing ManifestConsistencyError on lc207,
and the y*-matching KEEP_ACTIVE gap on lc208). Averages below are computed
per-variant over however many instances actually completed for that
variant, N is reported so a 10-instance average is never silently
presented as if it were the full 12.
"""
import csv
import matplotlib.pyplot as plt

SUMMARY_CSV = "outputs/pruner_comparison_summary/opt_mode_optimal_bi200_ss100_t0.5.csv"

with open(SUMMARY_CSV) as f:
    rows = list(csv.DictReader(f))

by_instance = {}
for r in rows:
    by_instance.setdefault(r["instance"], {})[r["variant"]] = r

VARIANTS = ["baseline", "request", "pair", "both"]
LABELS = {"baseline": "Baseline", "request": "Request Pruner", "pair": "Pair Pruner", "both": "Both"}
COLORS = {"baseline": "#9AA5A9", "request": "#276575", "pair": "#A6493A", "both": "#3E8F6C"}

avg_service_rate = {}
n_service_rate = {}
for v in VARIANTS:
    vals = [float(r[v]["service_rate_pct"]) for r in [by_instance[i] for i in by_instance] if r[v]["status"] == "ok"]
    avg_service_rate[v] = sum(vals) / len(vals)
    n_service_rate[v] = len(vals)

avg_runtime_reduction = {"baseline": 0.0}
n_runtime_reduction = {"baseline": len(by_instance)}
for v in ["request", "pair", "both"]:
    reductions = []
    for inst, variants in by_instance.items():
        base, cur = variants["baseline"], variants[v]
        if base["status"] != "ok" or cur["status"] != "ok":
            continue
        base_rt, cur_rt = float(base["runtime_seconds"]), float(cur["runtime_seconds"])
        reductions.append((base_rt - cur_rt) / base_rt * 100)
    avg_runtime_reduction[v] = sum(reductions) / len(reductions)
    n_runtime_reduction[v] = len(reductions)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("OPT (Solution Projection with Rolling Horizon) — Pruner-Vergleich, Ø über Testinstanzen (bi200/ss100, t=0.5)", fontsize=12.5)

x = range(len(VARIANTS))
labels = [f"{LABELS[v]}\n(n={n_runtime_reduction[v]})" for v in VARIANTS]
colors = [COLORS[v] for v in VARIANTS]

ax = axes[0]
vals = [avg_runtime_reduction[v] for v in VARIANTS]
bars = ax.bar(x, vals, color=colors)
ax.set_title("Laufzeitreduktion ggü. Baseline (%)")
ax.set_ylabel("Laufzeitreduktion ggü. Baseline (%)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3 if v >= 0 else -12), ha="center", fontsize=9)

ax = axes[1]
labels2 = [f"{LABELS[v]}\n(n={n_service_rate[v]})" for v in VARIANTS]
vals = [avg_service_rate[v] for v in VARIANTS]
bars = ax.bar(x, vals, color=colors)
ax.set_title("Service Rate (%)")
ax.set_ylabel("Service Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(list(x)); ax.set_xticklabels(labels2, fontsize=9)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out_path = "outputs/pruner_comparison_summary/fig_opt_avg_runtime_service_rate.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
print("avg_service_rate:", {k: round(v, 2) for k, v in avg_service_rate.items()}, n_service_rate)
print("avg_runtime_reduction:", {k: round(v, 2) for k, v in avg_runtime_reduction.items()}, n_runtime_reduction)
