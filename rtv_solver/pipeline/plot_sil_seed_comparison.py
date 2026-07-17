"""
2026-07-17: two clean bar-chart figures for RHO mit Structured Imitation
Learning (SIL) - runtime and service rate, both seeds (42, 1), all three
thresholds (0.4/0.5/0.6), bi400/ss200. Source: wall-clock timestamps from
the seed42/seed1 sweep master logs (runtime) and val/epoch_4 results.json
(service rate), both already collected earlier in the session.
"""
import matplotlib.pyplot as plt

VARIANTS = ["Baseline", "Request Pruner", "Pair Pruner", "Both"]
THRESHOLDS = ["0.4", "0.5", "0.6"]
SEED_COLORS = {"42": "#276575", "1": "#c17a3d"}

# runtime in minutes
RUNTIME = {
    "42": {
        "0.4": {"Baseline": 18.77, "Request Pruner": 14.97, "Pair Pruner": 12.92, "Both": 11.95},
        "0.5": {"Baseline": 18.77, "Request Pruner": 15.05, "Pair Pruner": 12.48, "Both": 11.25},
        "0.6": {"Baseline": 18.77, "Request Pruner": 13.77, "Pair Pruner": 12.47, "Both": 10.87},
    },
    "1": {
        "0.4": {"Baseline": 16.43, "Request Pruner": 15.33, "Pair Pruner": 13.08, "Both": 12.18},
        "0.5": {"Baseline": 16.43, "Request Pruner": 14.87, "Pair Pruner": 13.13, "Both": 11.88},
        "0.6": {"Baseline": 16.43, "Request Pruner": 14.40, "Pair Pruner": 12.88, "Both": 11.23},
    },
}

# service rate in %
SERVICE_RATE = {
    "42": {
        "0.4": {"Baseline": 54.08, "Request Pruner": 50.9, "Pair Pruner": 54.1, "Both": 50.6},
        "0.5": {"Baseline": 54.08, "Request Pruner": 49.1, "Pair Pruner": 54.1, "Both": 48.4},
        "0.6": {"Baseline": 54.08, "Request Pruner": 44.0, "Pair Pruner": 54.1, "Both": 44.0},
    },
    "1": {
        "0.4": {"Baseline": 50.65, "Request Pruner": 49.08, "Pair Pruner": 50.65, "Both": 48.76},
        "0.5": {"Baseline": 50.65, "Request Pruner": 40.60, "Pair Pruner": 50.65, "Both": 39.65},
        "0.6": {"Baseline": 50.65, "Request Pruner": 30.52, "Pair Pruner": 50.95, "Both": 31.79},
    },
}


def make_figure(data, ylabel, title, out_path, ylim=None, fmt="{:.1f}"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"RHO mit Structured Imitation Learning (SIL) — {title}\nbi400/ss200, Seed 42 vs. Seed 1", fontsize=13)

    x = range(len(VARIANTS))
    bar_width = 0.35

    for ax, th in zip(axes, THRESHOLDS):
        for i, seed in enumerate(["42", "1"]):
            vals = [data[seed][th][v] for v in VARIANTS]
            xs = [xi + (i - 0.5) * bar_width for xi in x]
            bars = ax.bar(xs, vals, width=bar_width, color=SEED_COLORS[seed], label=f"Seed {seed}")
            for b, v in zip(bars, vals):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
        ax.set_title(f"Threshold {th}")
        ax.set_xticks(list(x))
        ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=9, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


make_figure(
    RUNTIME, "Laufzeit (Minuten, kompletter 5-Epochen-Trainingslauf)",
    "Laufzeit-Vergleich",
    "outputs/pruner_comparison_summary/fig_sil_seed_comparison_runtime.png",
)

make_figure(
    SERVICE_RATE, "Service Rate (%)",
    "Service-Rate-Vergleich",
    "outputs/pruner_comparison_summary/fig_sil_seed_comparison_service_rate.png",
    ylim=(0, 100),
)
