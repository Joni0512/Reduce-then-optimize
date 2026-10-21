"""
2026-09-27: c3, class1, Combined Pruner (Pair Pruner fixed at threshold=0.5), seed=42 -
Request Pruner threshold swept over {0.5, 0.7, 0.8, 0.9}. Same trained MLP checkpoint each
time, only the decision threshold changes - so this isolates the effect of pruning AMOUNT
on service rate, without the confound of different training runs (unlike the earlier
cross-run pruning-rate-vs-service-rate scatter). Shows each config has its own "cliff"
pruning rate beyond which service rate collapses.

Source (all "Best val service rate" + compute_pruning_rates.py, 2026-09-26/27):
  sil_training_{cfg}_c3_class1_combined_reqprefix_pairv1final_pairt0.5_seed42.log (t=0.5)
  sil_training_{cfg}_c3_class1_combined_reqprefix_reqt{0.7,0.8,0.9}_pairv1final_pairt0.5_seed42.log

Run: ./venv/bin/python3 scripts/analysis/make_c3_threshold_sweep_line_chart.py
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "figures_export/Request Pruner SIL"

# (threshold, request_pruning_rate_%, service_rate_%)
DATA = {
    "batch_interval=200/step_size=100": [(0.5, 0.00, 86.16), (0.7, 1.84, 87.11), (0.8, 13.75, 72.64), (0.9, 28.19, 46.86)],
    "batch_interval=400/step_size=200": [(0.5, 1.08, 68.55), (0.7, 5.92, 69.81), (0.8, 8.97, 69.81), (0.9, 14.67, 66.04)],
    "batch_interval=400/step_size=80": [(0.5, 0.35, 87.74), (0.7, 7.11, 88.99), (0.8, 18.45, 88.99), (0.9, 31.24, 66.35)],
}
COLORS = {"batch_interval=200/step_size=100": "#276575", "batch_interval=400/step_size=200": "#A6493A", "batch_interval=400/step_size=80": "#3E8F6C"}


def make_chart():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for cfg, points in DATA.items():
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ts = [p[0] for p in points]
        ax.plot(xs, ys, marker="o", color=COLORS[cfg], linewidth=2, markersize=7, label=cfg, zorder=3)
        for t, x, y in zip(ts, xs, ys):
            ax.annotate(f"t={t}", (x, y), textcoords="offset points", xytext=(6, 6),
                        fontsize=8.5, color=COLORS[cfg])

    ax.set_xlabel("Request Pruner: live pruning rate (%)", fontsize=11)
    ax.set_ylabel("Best val service rate (%)", fontsize=11)
    ax.set_ylim(35, 95)
    ax.set_xlim(-2, 35)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("c3 Request Pruner threshold sweep (Combined, Pair Pruner fixed at $t{=}0.5$)\n"
                 "same trained checkpoint, only decision threshold varies - seed=42, class1",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=10, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    fig.savefig(f"{OUT_DIR}/c3_threshold_sweep_pruning_vs_service_rate.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/c3_threshold_sweep_pruning_vs_service_rate.pdf")
    print(f"saved {OUT_DIR}/c3_threshold_sweep_pruning_vs_service_rate.{{png,pdf}}")


if __name__ == "__main__":
    make_chart()
