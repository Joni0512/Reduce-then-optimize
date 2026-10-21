"""
2026-09-26: c3 (max_cardinality=3), class1, Baseline (no pruner) vs. Request Pruner
(expert-prefix-label, k=3, threshold=0.5, min_retention_fraction=0.3) - grouped bar chart
per config, with individual seed values overlaid as dots (only 1-2 seeds per cell so far,
not enough for a meaningful std/error bar - see chat 2026-09-26).

Best val service rate values, verified via "Best val service rate" grep on:
  sil_training_bi200_ss100_c3_class1_legacy.log (seed42), ..._legacy_seed1.log (seed1)
  sil_training_bi400_ss200_c3_class1_legacy_seed{42,1}.log
  sil_training_bi400_ss80_c3_class1_legacy_seed1.log
  (seed42 baseline for bi400/ss80 missing - local log incomplete, never finished)
  sil_training_{cfg}_c3_class1_requestpruner_prefixlabel_t0.5_seed{42,1}.log

Run: ./venv/bin/python3 scripts/analysis/make_c3_class1_baseline_vs_pruner_chart.py
"""
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

CONFIGS = ["bi200/ss100", "bi400/ss200", "bi400/ss80"]

# per-seed raw values (None = missing)
BASELINE = {
    "bi200/ss100": {42: 86.79, 1: 87.11},
    "bi400/ss200": {42: 68.55, 1: 70.44},
    "bi400/ss80": {42: None, 1: 88.99},
}
PRUNER = {
    "bi200/ss100": {42: 86.48, 1: 87.11},
    "bi400/ss200": {42: 70.13, 1: 71.07},
    "bi400/ss80": {42: 88.68, 1: 89.31},
}

COLOR_BASELINE = "#1E2761"
COLOR_PRUNER = "#276575"
SEED_MARKER_COLOR = "#E0A030"


def mean_of(d):
    vals = [v for v in d.values() if v is not None]
    return float(np.mean(vals)), len(vals)


def make_chart():
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(CONFIGS))
    width = 0.32

    base_means, base_ns = [], []
    pruner_means, pruner_ns = [], []
    for cfg in CONFIGS:
        m, n = mean_of(BASELINE[cfg])
        base_means.append(m)
        base_ns.append(n)
        m, n = mean_of(PRUNER[cfg])
        pruner_means.append(m)
        pruner_ns.append(n)

    bars_base = ax.bar(x - width / 2, base_means, width=width, color=COLOR_BASELINE,
                        zorder=3, label="No pruner (baseline)")
    bars_pruner = ax.bar(x + width / 2, pruner_means, width=width, color=COLOR_PRUNER,
                          zorder=3, label="Request Pruner (prefix, $k{=}3$, $t{=}0.5$)")

    # overlay individual seed dots
    for xi, cfg in zip(x, CONFIGS):
        for seed, val in BASELINE[cfg].items():
            if val is not None:
                ax.scatter(xi - width / 2, val, color=SEED_MARKER_COLOR, edgecolor="black",
                           zorder=5, s=45)
        for seed, val in PRUNER[cfg].items():
            if val is not None:
                ax.scatter(xi + width / 2, val, color=SEED_MARKER_COLOR, edgecolor="black",
                           zorder=5, s=45)

    # bar value labels + n
    for xi, m, n in zip(x - width / 2, base_means, base_ns):
        ax.annotate(f"{m:.2f}%\n(n={n})", (xi, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9.5, fontweight="bold")
    for xi, m, n in zip(x + width / 2, pruner_means, pruner_ns):
        ax.annotate(f"{m:.2f}%\n(n={n})", (xi, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9.5, fontweight="bold")

    # delta annotation between bars
    for xi, cfg, bm, pm in zip(x, CONFIGS, base_means, pruner_means):
        delta = pm - bm
        sign = "+" if delta >= 0 else "−"
        ax.annotate(f"$\\Delta$ {sign}{abs(delta):.2f}pp", (xi, max(bm, pm) + 6),
                    ha="center", fontsize=9, color="#444444")

    ax.scatter([], [], color=SEED_MARKER_COLOR, edgecolor="black", label="individual seed")

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=11)
    ax.set_ylabel("Best val service rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("c3 (max_cardinality=3), class1: Baseline vs. Request Pruner\n"
                 "epochs=5, lr=0.0001, legacy scoring", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(0.5, -0.16, "bi400/ss80 baseline seed42 missing (log never completed) - only n=1 there",
            transform=ax.transAxes, ha="center", fontsize=8, color="#888888")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/c3_class1_baseline_vs_pruner_service_rate.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/c3_class1_baseline_vs_pruner_service_rate.pdf")
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/c3_class1_baseline_vs_pruner_service_rate.{{png,pdf}}")


if __name__ == "__main__":
    make_chart()
