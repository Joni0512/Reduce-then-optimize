"""
2026-09-25: Baseline vs. Request-Pruner-only vs. Request+Pair-Pruner comparison for
bi400/ss200 MIXED-CLASS (threshold=0.5, min_retention_fraction=0.3), analog to
make_bi400ss200_combined_pruner_chart.py (class1 version).

Sources (2026-09-23/24/25, "Best val service rate" + start/"Run complete" timestamps):
- baseline, 5 seeds: sil_training_bi400_ss200_mixedclass_requestpruner_baseline_seed{...}.log
  -> SR 43.50/47.51/47.99/45.10/46.71 (mean 46.16, std 1.85); RT 184.1/170.1/184.8/182.8/183.5 min (mean 181.1, std 6.2)
- request-only, 5 seeds: sil_training_bi400_ss200_mixedclass_requestpruner_prefixlabel_t0.5_seed{...}.log
  -> SR 49.44/49.28/48.64/44.78/51.85 (mean 48.80, std 2.56); RT 101.0/102.0/101.1/101.6/105.8 min (mean 102.3, std 2.0)
- combined, 1 seed only (42): sil_training_bi400_ss200_mixedclass_combined_reqprefix_pairv1final_pairt0.5_seed42.log
  -> SR 47.67; RT 62.4 min (sacct elapsed 01:02:23)

Run: python3 -m scripts.analysis.make_bi400ss200_mixedclass_combined_pruner_chart
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"

VARIANTS = ["baseline", "request", "combined"]
LABELS = ["No pruner\n(baseline)", "Request Pruner\nonly", "Request + Pair\nPruner"]
COLORS = ["#1E2761", "#276575", "#3E8F6C"]

SERVICE_MEAN = {"baseline": 46.16, "request": 48.80, "combined": 47.67}
SERVICE_STD = {"baseline": 1.85, "request": 2.56, "combined": None}

RUNTIME_MEAN = {"baseline": 181.1, "request": 102.3, "combined": 62.4}
RUNTIME_STD = {"baseline": 6.2, "request": 2.0, "combined": None}

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
    ax.bar(x, vals, yerr=errs, capsize=5, color=COLORS, width=0.6, zorder=3)

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
    subtitle = "bi400/ss200, MIXED-CLASS, n=5 seeds (baseline/request), n=1 (combined, seed 42)\nthreshold=0.5, min_retention_fraction=0.3"
    if note:
        subtitle += f"\n{note}"
    ax.set_title(f"{subtitle}\n{title}", fontsize=11)
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
               "bi400ss200_mixedclass_combined_pruner_service_rate", ylim=(0, 100), pct=True)
    make_chart(RUNTIME_MEAN, RUNTIME_STD, "Runtime (minutes, full 5-epoch training run)", "Runtime",
               "bi400ss200_mixedclass_combined_pruner_runtime", ylim=(0, 220),
               note="combined = single seed (42) only, not yet replicated")
