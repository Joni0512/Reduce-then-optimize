"""
2026-07-16: RHO ohne Learning, threshold sweep 0.4/0.5/0.6, averaged over the
12 test instances. Same visual structure as the reference chart the user
pasted (grouped bars, two side-by-side panels: runtime reduction vs.
baseline (%), and service rate (%)) - x-axis groups are the three pruner
variants (Request / Pair / Both), each with 3 threshold sub-bars, matching
the reference's "category on x, method+threshold as sub-bars" layout.
Source: outputs/pruner_comparison_summary/rho_no_learning_threshold_summary.csv
(built by aggregating the three run_rho_pruner_comparison_12*.sh sweeps).
"""
import csv
import matplotlib.pyplot as plt

SUMMARY_CSV = "outputs/pruner_comparison_summary/rho_no_learning_threshold_summary.csv"

with open(SUMMARY_CSV) as f:
    rows = list(csv.DictReader(f))

baseline_row = next(r for r in rows if r["variant"] == "baseline")
variant_rows = {(r["variant"], r["threshold"]): r for r in rows if r["variant"] != "baseline"}

VARIANTS = ["request", "pair", "both"]
VARIANT_LABELS = {"request": "Request Pruner", "pair": "Pair Pruner", "both": "Both"}
THRESHOLDS = ["0.4", "0.5", "0.6"]
THRESHOLD_COLORS = {"0.4": "#a9cce3", "0.5": "#2e86ab", "0.6": "#0b3d5c"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("RHO ohne Learning — Threshold-Vergleich 0.4 / 0.5 / 0.6 (Ø über 12 Testinstanzen, bi200/ss100)", fontsize=13)

n_groups = len(VARIANTS)
n_bars = len(THRESHOLDS)
bar_width = 0.8 / n_bars
group_centers = range(n_groups)

def plot_panel(ax, value_key, ylabel, title, baseline_value, ylim=None):
    for j, th in enumerate(THRESHOLDS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [float(variant_rows[(v, th)][value_key]) for v in VARIANTS]
        bars = ax.bar(xs, vals, width=bar_width, color=THRESHOLD_COLORS[th], label=f"t={th}")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2 if v >= 0 else -12),
                        ha="center", fontsize=8)
    if baseline_value is not None:
        ax.axhline(baseline_value, color="gray", linewidth=1, linestyle="--", label="Baseline")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, ncol=2)

plot_panel(
    axes[0], "avg_runtime_reduction_pct",
    "Laufzeitreduktion ggü. Baseline (%)", "Laufzeitreduktion: Threshold 0.4 / 0.5 / 0.6",
    baseline_value=0.0,
)
plot_panel(
    axes[1], "avg_service_rate_pct",
    "Service Rate (%)", "Service Rate: Threshold 0.4 / 0.5 / 0.6",
    baseline_value=float(baseline_row["avg_service_rate_pct"]), ylim=(0, 100),
)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out_path = "outputs/pruner_comparison_summary/fig_rho_threshold_sweep.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
