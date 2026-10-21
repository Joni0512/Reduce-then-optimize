"""
2026-09-24: Baseline vs. Request-Pruner-only vs. Request+Pair-Pruner comparison for
bi400/ss200 class1 (threshold=0.5, min_retention_fraction=0.3), analog to
make_bi200ss100_combined_pruner_chart.py.

All service rates use "Best val service rate" (the saved/deployed checkpoint), not a
fixed epoch - confirmed as the consistent convention across this whole comparison
(2026-09-24 chat: a stale 57.23% for combined-seed42 in the user's own draft text turned
out to be epoch 2/5's raw value, not the best-val checkpoint at epoch 3 = 57.55%).

Sources (2026-09-24, "Best val service rate" lines):
- request-only, 5 seeds (42,1,2,3,4): sil_training_bi400_ss200_class1_requestpruner_prefixlabel_t0.5_seed{...}.log
  -> 57.86, 53.46, 55.97, 55.66, 56.60 (mean 55.91, std 1.66); runtime 68.98/70.02/66.65/66.70/67.0 min (mean 67.9, std 1.6)
- combined, 5 seeds: sil_training_bi400_ss200_class1_combined_reqprefix_pairv1final_pairt0.5_seed{...}.log
  -> 57.55, 54.72, 55.97, 55.66, 57.23 (mean 56.23, std 1.16)
  runtime: seed42 ran LOCALLY (8.8min, not comparable - different HW/threading), seeds 1-4 ran on the
  cluster (38.8/36.2/35.0/35.9 min, mean 36.5, std 1.6) - only the 4 cluster seeds are used for the
  runtime bar to keep it apples-to-apples with the baseline/request-only cluster runtimes.
- baseline (no pruner): 51.8+-1.6% service rate (request_pruner_deck.pptx table, 5-seed, n=5 stated in
  that slide's subtitle - not independently re-verified from raw logs, none found locally for this
  config). Runtime 79.3min is the user's own cited figure (source/seed-count not independently verified
  by me - no local baseline log for bi400/ss200 class1 exists to check against).

Run: python3 -m scripts.analysis.make_bi400ss200_combined_pruner_chart
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

VARIANTS = ["baseline", "request", "combined"]
LABELS = ["No pruner\n(baseline)", "Request Pruner\nonly", "Request + Pair\nPruner"]
COLORS = ["#1E2761", "#276575", "#3E8F6C"]

SERVICE_MEAN = {"baseline": 51.8, "request": 55.91, "combined": 56.23}
SERVICE_STD = {"baseline": 1.6, "request": 1.66, "combined": 1.16}

RUNTIME_MEAN = {"baseline": 79.3, "request": 67.9, "combined": 36.5}
RUNTIME_STD = {"baseline": None, "request": 1.6, "combined": 1.6}  # baseline: user-cited figure, no per-seed std available

HP_ANNOTATIONS = {
    "baseline": "no pruner",
    "request": "Req: h32/l1/d0.0\npw1.0/lr0.03, t=0.5",
    "combined": "Req: h32/l1/d0.0, pw1.0/lr0.03, t=0.5\nPair: GNNv1 h64/mp2, pw2.0/lr0.001, t=0.5",
}


def make_chart(values, stds, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None, pct=False, note=None):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.arange(len(VARIANTS))
    vals = [values[v] for v in VARIANTS]
    errs = [stds[v] if stds[v] is not None else 0 for v in VARIANTS]
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=COLORS, width=0.6, zorder=3)

    for xi, v, s, var in zip(x, vals, errs, VARIANTS):
        label = value_fmt.format(v) if stds[var] is None else f"{value_fmt.format(v)} ± {value_fmt.format(s)}"
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
    subtitle = "bi400/ss200, class1, n=5 seeds (42,1,2,3,4)\nthreshold=0.5, min_retention_fraction=0.3"
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
               "bi400ss200_class1_combined_pruner_service_rate", ylim=(0, 100), pct=True)
    make_chart(RUNTIME_MEAN, RUNTIME_STD, "Runtime (minutes, full 5-epoch training run)", "Runtime",
               "bi400ss200_class1_combined_pruner_runtime", ylim=(0, 95),
               note="combined runtime = 4 cluster seeds only (seed42 ran locally, not comparable, see script docstring)")
