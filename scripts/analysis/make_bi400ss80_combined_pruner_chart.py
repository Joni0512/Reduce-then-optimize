"""
2026-09-25: Baseline vs. Request-Pruner-only vs. Request+Pair-Pruner comparison for
bi400/ss80 class1 (threshold=0.5, min_retention_fraction=0.3), analog to
make_bi200ss100_combined_pruner_chart.py / make_bi400ss200_combined_pruner_chart.py.

All service rates use "Best val service rate" (saved/deployed checkpoint).

Sources (2026-09-24/25):
- baseline, n=4 seeds (42,1,2,3; seed4 pending): sil_training_bi400_ss80_class1_baseline_seed{...}.log
  -> 79.87, 78.93, 78.62, 79.87 (mean 79.32, std 0.64); runtime 106.8/109.8/102.2/111.1 min (mean 107.5, std 4.0)
- request-only, 5 seeds: sil_training_bi400_ss80_class1_requestpruner_prefixlabel_t0.5_seed{...}.log
  -> 82.70, 82.08, 82.70, 83.65, 81.76 (mean 82.58, std 0.72); runtime 65.7/63.8/63.8/63.1/63.1 min (mean 63.9, std 1.1)
- combined, 5 seeds: sil_training_bi400_ss80_class1_combined_reqprefix_pairv1final_pairt0.5_seed{...}.log
  -> 81.45, 82.08, 83.02, 83.33, 81.13 (mean 82.20, std 0.96)
  runtime: seed42 ran LOCALLY (10.2min, not comparable), seeds 1-4 on cluster (43.6/46.6/42.7/45.0 min,
  mean 44.5, std 1.7) - only the 4 cluster seeds used for the runtime bar, same convention as the other
  2 configs' charts.

Run: python3 -m scripts.analysis.make_bi400ss80_combined_pruner_chart
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

VARIANTS = ["baseline", "request", "combined"]
LABELS = ["No pruner\n(baseline)", "Request Pruner\nonly", "Request + Pair\nPruner"]
COLORS = ["#1E2761", "#276575", "#3E8F6C"]

SERVICE_MEAN = {"baseline": 79.32, "request": 82.58, "combined": 82.20}
SERVICE_STD = {"baseline": 0.64, "request": 0.72, "combined": 0.96}

RUNTIME_MEAN = {"baseline": 107.5, "request": 63.9, "combined": 44.5}
RUNTIME_STD = {"baseline": 4.0, "request": 1.1, "combined": 1.7}

HP_ANNOTATIONS = {
    "baseline": "no pruner",
    "request": "Req: h64/l1/d0.0\npw1.5/lr0.03, t=0.5",
    "combined": "Req: h64/l1/d0.0, pw1.5/lr0.03, t=0.5\nPair: GNNv1 h64/mp2, pw2.0/lr0.001, t=0.5",
}


def make_chart(values, stds, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None, pct=False, note=None):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.arange(len(VARIANTS))
    vals = [values[v] for v in VARIANTS]
    errs = [stds[v] if stds[v] is not None else 0 for v in VARIANTS]
    ax.bar(x, vals, yerr=errs, capsize=5, color=COLORS, width=0.6, zorder=3)

    for xi, v, s, var in zip(x, vals, errs, VARIANTS):
        label = f"{value_fmt.format(v)} ± {value_fmt.format(s)}"
        ax.annotate(label, (xi, v + s), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=11, fontweight="bold")
        ax.annotate(HP_ANNOTATIONS[var], (xi, max(vals) * 1.12 if not ylim else ylim[1] * 0.9),
                    ha="center", fontsize=8.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    subtitle = "bi400/ss80, class1, n=4-5 seeds (42,1,2,3[,4])\nthreshold=0.5, min_retention_fraction=0.3"
    if note:
        subtitle += f"\n{note}"
    ax.set_title(f"{subtitle}\n{title}", fontsize=11.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{out_stem}.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/{out_stem}.pdf")
    print(f"saved {OUT_DIR}/{out_stem}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_chart(SERVICE_MEAN, SERVICE_STD, "Best val service rate (%)", "Service Rate",
               "bi400ss80_class1_combined_pruner_service_rate", ylim=(0, 100), pct=True)
    make_chart(RUNTIME_MEAN, RUNTIME_STD, "Runtime (minutes, full 5-epoch training run)", "Runtime",
               "bi400ss80_class1_combined_pruner_runtime", ylim=(0, 130),
               note="combined runtime = 4 cluster seeds only (seed42 ran locally, not comparable)")
