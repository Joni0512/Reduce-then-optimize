"""
2026-07-18: two English figures comparing all 4 tested window configs for
RHO mit Structured Imitation Learning (SIL) - service rate and runtime,
threshold 0.5, seed 42, 4 pruner variants each. Source: val/epoch_4
results.json (service rate) and master-log timestamps (runtime), already
collected earlier in the session for each config's pipeline run.
"""
import matplotlib.pyplot as plt

CONFIGS = ["bi200/ss40", "bi200/ss100", "bi400/ss80", "bi400/ss200"]
VARIANTS = ["Baseline", "Request Pruner", "Pair Pruner", "Both"]
VARIANT_COLORS = {"Baseline": "#9AA5A9", "Request Pruner": "#276575", "Pair Pruner": "#A6493A", "Both": "#3E8F6C"}

SERVICE_RATE = {
    "bi200/ss40":  {"Baseline": 84.27, "Request Pruner": 32.38, "Pair Pruner": 85.22, "Both": 32.38},
    "bi200/ss100": {"Baseline": 75.52, "Request Pruner": 43.97, "Pair Pruner": 75.52, "Both": 43.97},
    "bi400/ss80":  {"Baseline": 75.18, "Request Pruner": 60.32, "Pair Pruner": 75.48, "Both": 59.68},
    "bi400/ss200": {"Baseline": 54.08, "Request Pruner": 49.10, "Pair Pruner": 54.10, "Both": 48.40},
}

RUNTIME = {
    "bi200/ss40":  {"Baseline": 32.70, "Request Pruner": 21.85, "Pair Pruner": 26.78, "Both": 19.92},
    "bi200/ss100": {"Baseline": 19.58, "Request Pruner": 11.75, "Pair Pruner": 15.55, "Both": 10.85},
    "bi400/ss80":  {"Baseline": 23.98, "Request Pruner": 28.93, "Pair Pruner": 18.63, "Both": 16.70},
    "bi400/ss200": {"Baseline": 18.77, "Request Pruner": 15.05, "Pair Pruner": 12.48, "Both": 11.25},
}


def make_figure(data, ylabel, title, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"RHO with Structured Imitation Learning (SIL) — {title}\n"
                 "All 4 window configs tested, threshold 0.5, seed 42", fontsize=12.5)

    n_groups = len(CONFIGS)
    n_bars = len(VARIANTS)
    bar_width = 0.8 / n_bars
    group_centers = range(n_groups)

    for j, variant in enumerate(VARIANTS):
        xs = [g - 0.4 + bar_width * j + bar_width / 2 for g in group_centers]
        vals = [data[cfg][variant] for cfg in CONFIGS]
        bars = ax.bar(xs, vals, width=bar_width, color=VARIANT_COLORS[variant], label=variant)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)

    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_centers))
    ax.set_xticklabels(CONFIGS, fontsize=10)
    ax.set_xlabel("Window config (batch_interval / step_size)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


make_figure(
    SERVICE_RATE, "Service Rate (%)", "Service Rate by Window Config",
    "outputs/pruner_comparison_summary/fig_sil_4window_service_rate.png",
    ylim=(0, 100),
)
make_figure(
    RUNTIME, "Runtime (minutes, full 5-epoch training run)", "Runtime by Window Config",
    "outputs/pruner_comparison_summary/fig_sil_4window_runtime.png",
)
