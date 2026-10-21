"""
2026-09-24: Baseline vs. Request-Pruner-only vs. Request+Pair-Pruner comparison for
bi200/ss100 class1 (threshold=0.5, min_retention_fraction=0.3), n=5 seeds (42,1,2,3,4).

Service rate values verified directly from "Best val service rate" lines in
sil_training_bi200_ss100_class1_requestpruner_prefixlabel_t0.5_seed{42,1,2,3,4}.log
and sil_training_bi200_ss100_class1_combined_reqprefix_pairv1final_pairt0.5_seed{...}.log
(2026-09-24, corrected the request_pruner_deck.pptx table's stale 60.9±4.5% -> 65.85±2.45%
for the request-pruner-only bar, which did not match a fresh recompute from these logs).
Baseline (no pruner) service rate 74.5±1.8% and runtime ~95.6min (5-seed mean) are the
pre-established values already used throughout this comparison (deck slide 5 table;
95.6min underlies the 49.4%/63.0% runtime-saving figures already cited for seed 42) - no
per-seed baseline runtime std available locally (only seed42/seed1 raw logs synced).

Run: python3 -m scripts.analysis.make_bi200ss100_combined_pruner_chart
    (or: ./venv/bin/python3 scripts/analysis/make_bi200ss100_combined_pruner_chart.py)
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

VARIANTS = ["baseline", "request", "combined"]
LABELS = ["No pruner\n(baseline)", "Request Pruner\nonly", "Request + Pair\nPruner"]
COLORS = ["#1E2761", "#276575", "#3E8F6C"]

SERVICE_MEAN = {"baseline": 74.5, "request": 65.85, "combined": 64.40}
SERVICE_STD = {"baseline": 1.8, "request": 2.45, "combined": 3.57}

RUNTIME_MEAN = {"baseline": 95.6, "request": 49.4, "combined": 37.0}
RUNTIME_STD = {"baseline": None, "request": 1.4, "combined": 1.7}  # baseline: no per-seed std available locally

HP_ANNOTATIONS = {
    "baseline": "no pruner",
    "request": "Req: h32/l1/d0.2\npw1.0/lr0.01, t=0.5",
    "combined": "Req: h32/l1/d0.2, pw1.0/lr0.01, t=0.5\nPair: GNNv1 h64/mp2, pw2.0/lr0.001, t=0.5",
}


def make_chart(values, stds, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None, pct=False):
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
    ax.set_title(f"bi200/ss100, class1, n=5 seeds (42,1,2,3,4)\nthreshold=0.5, min_retention_fraction=0.3\n{title}",
                 fontsize=12)
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
               "bi200ss100_class1_combined_pruner_service_rate", ylim=(0, 100), pct=True)
    make_chart(RUNTIME_MEAN, RUNTIME_STD, "Runtime (minutes, full 5-epoch training run)", "Runtime",
               "bi200ss100_class1_combined_pruner_runtime", ylim=(0, 115))
