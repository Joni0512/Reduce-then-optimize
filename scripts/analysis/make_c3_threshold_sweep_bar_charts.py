"""
2026-09-27: c3, class1, Combined Pruner (Pair Pruner fixed at threshold=0.5), seed=42 -
grouped bar charts (Service Rate, Runtime) across the Request Pruner threshold sweep
{0.5, 0.7, 0.8, 0.9}, one panel per config. Companion to
make_c3_threshold_sweep_line_chart.py (pruning-rate-vs-service-rate line chart).

Source: same "Best val service rate" logs as the line chart, plus log-timestamp-derived
runtimes (start of first log line to last log line, 2026-09-27, see chat).

Run: ./venv/bin/python3 scripts/analysis/make_c3_threshold_sweep_bar_charts.py
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
THRESHOLDS = ["0.5", "0.7", "0.8", "0.9"]
COLORS = ["#1E2761", "#276575", "#E8A33D", "#A6493A"]

SERVICE_RATE = {
    "batch_interval=200/step_size=100": {"0.5": 86.16, "0.7": 87.11, "0.8": 72.64, "0.9": 46.86},
    "batch_interval=400/step_size=200": {"0.5": 68.55, "0.7": 69.81, "0.8": 69.81, "0.9": 66.04},
    "batch_interval=400/step_size=80": {"0.5": 87.74, "0.7": 88.99, "0.8": 88.99, "0.9": 66.35},
}
# runtime in minutes (log-timestamp start-to-end, seed=42)
RUNTIME_MIN = {
    "batch_interval=200/step_size=100": {"0.5": 132.8, "0.7": 134.5, "0.8": 87.6, "0.9": 49.5},
    "batch_interval=400/step_size=200": {"0.5": 133.0, "0.7": 107.6, "0.8": 105.0, "0.9": 99.2},
    "batch_interval=400/step_size=80": {"0.5": 148.4, "0.7": 149.6, "0.8": 102.0, "0.9": 63.7},
}


def make_grouped_bars(data, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None, pct=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        x = np.arange(len(THRESHOLDS))
        vals = [data[cfg][t] for t in THRESHOLDS]
        bars = ax.bar(x, vals, color=COLORS, width=0.6, zorder=3)
        for xi, v in zip(x, vals):
            ax.annotate(value_fmt.format(v), (xi, v), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=9.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t={t}" for t in THRESHOLDS], fontsize=10)
        ax.set_title(cfg, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{out_stem}.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/{out_stem}.pdf")
    print(f"saved {OUT_DIR}/{out_stem}.{{png,pdf}}")


if __name__ == "__main__":
    make_grouped_bars(SERVICE_RATE, "Best val service rate (%)",
                       "c3 Request Pruner threshold sweep: Service Rate\n"
                       "Combined Pruner, Pair Pruner fixed at $t{=}0.5$, seed=42, class1",
                       "c3_threshold_sweep_service_rate_bars", ylim=(0, 100), pct=True)
    make_grouped_bars(RUNTIME_MIN, "Runtime (minutes, full 5-epoch training run)",
                       "c3 Request Pruner threshold sweep: Runtime\n"
                       "Combined Pruner, Pair Pruner fixed at $t{=}0.5$, seed=42, class1",
                       "c3_threshold_sweep_runtime_bars", ylim=(0, 165))
