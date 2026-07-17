"""
2026-07-17: OPT (Solution Projection with Rolling Horizon) (COAML mode="optimal"),
threshold sweep 0.4 / 0.5 / 0.6, averaged over the test instances -
counterpart to plot_rho_threshold_sweep.py (RHO ohne Learning) - note: this is OPT, not SIL, see chat correction. Reads the
three per-threshold CSVs already saved under outputs/pruner_comparison_summary/.

Averages are computed per-variant over however many instances completed
for that (variant, threshold) combination - lc207 crashes on baseline/pair
at every threshold (pre-existing, unrelated bug), lc208 crashes on
request/both only at t=0.5 (the y*-matching KEEP_ACTIVE gap), so N differs
by column and is reported on the chart instead of silently averaging over
a shrinking, inconsistent instance set.
"""
import csv
import matplotlib.pyplot as plt

FILES = {
    "0.4": "outputs/pruner_comparison_summary/opt_mode_optimal_bi200_ss100_t0.4.csv",
    "0.5": "outputs/pruner_comparison_summary/opt_mode_optimal_bi200_ss100_t0.5.csv",
    "0.6": "outputs/pruner_comparison_summary/opt_mode_optimal_bi200_ss100_t0.6.csv",
}

by_threshold = {}
for th, path in FILES.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_instance = {}
    for r in rows:
        by_instance.setdefault(r["instance"], {})[r["variant"]] = r
    by_threshold[th] = by_instance

VARIANTS = ["request", "pair", "both"]
LABELS = {"request": "Request Pruner", "pair": "Pair Pruner", "both": "Both"}
THRESHOLDS = ["0.4", "0.5", "0.6"]
THRESHOLD_COLORS = {"0.4": "#a9cce3", "0.5": "#2e86ab", "0.6": "#0b3d5c"}

avg_service_rate, n_service_rate = {}, {}
avg_runtime_reduction, n_runtime_reduction = {}, {}
avg_baseline_sr = {}

for th in THRESHOLDS:
    by_instance = by_threshold[th]
    base_vals = [float(r["baseline"]["service_rate_pct"]) for r in by_instance.values() if r["baseline"]["status"] == "ok"]
    avg_baseline_sr[th] = sum(base_vals) / len(base_vals)
    for v in VARIANTS:
        sr_vals = [float(r[v]["service_rate_pct"]) for r in by_instance.values() if r[v]["status"] == "ok"]
        avg_service_rate[(v, th)] = sum(sr_vals) / len(sr_vals)
        n_service_rate[(v, th)] = len(sr_vals)

        reductions = []
        for inst, variants in by_instance.items():
            base, cur = variants["baseline"], variants[v]
            if base["status"] != "ok" or cur["status"] != "ok":
                continue
            base_rt, cur_rt = float(base["runtime_seconds"]), float(cur["runtime_seconds"])
            reductions.append((base_rt - cur_rt) / base_rt * 100)
        avg_runtime_reduction[(v, th)] = sum(reductions) / len(reductions)
        n_runtime_reduction[(v, th)] = len(reductions)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("OPT (Solution Projection with Rolling Horizon) — Threshold-Vergleich 0.4 / 0.5 / 0.6 (Ø über Testinstanzen, bi200/ss100)", fontsize=12.5)

n_groups = len(VARIANTS)
n_bars = len(THRESHOLDS)
bar_width = 0.8 / n_bars
group_centers = range(n_groups)

def plot_panel(ax, get_val, get_n, ylabel, title, baseline_value=None, ylim=None):
    for j, th in enumerate(THRESHOLDS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [get_val(v, th) for v in VARIANTS]
        bars = ax.bar(xs, vals, width=bar_width, color=THRESHOLD_COLORS[th], label=f"t={th}")
        for b, v, vv in zip(bars, vals, VARIANTS):
            n = get_n(vv, th)
            ax.annotate(f"{v:.0f}\n(n={n})" if abs(v) >= 10 else f"{v:.1f}\n(n={n})",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2 if v >= 0 else -26),
                        ha="center", fontsize=7, rotation=0)
    if baseline_value is not None:
        ax.axhline(baseline_value, color="gray", linewidth=1, linestyle="--", label=f"Baseline (t=0.5, {baseline_value:.0f}%)")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels([LABELS[v] for v in VARIANTS])
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)

plot_panel(
    axes[0], lambda v, th: avg_runtime_reduction[(v, th)], lambda v, th: n_runtime_reduction[(v, th)],
    "Laufzeitreduktion ggü. Baseline (%)", "Laufzeitreduktion: Threshold 0.4 / 0.5 / 0.6",
)
plot_panel(
    axes[1], lambda v, th: avg_service_rate[(v, th)], lambda v, th: n_service_rate[(v, th)],
    "Service Rate (%)", "Service Rate: Threshold 0.4 / 0.5 / 0.6",
    baseline_value=avg_baseline_sr["0.5"], ylim=(0, 100),
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out_path = "outputs/pruner_comparison_summary/fig_opt_threshold_sweep.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
for k in sorted(avg_service_rate):
    print(k, "service_rate=", round(avg_service_rate[k],1), "n=", n_service_rate[k],
          "runtime_reduction=", round(avg_runtime_reduction[k],1), "n=", n_runtime_reduction[k])
