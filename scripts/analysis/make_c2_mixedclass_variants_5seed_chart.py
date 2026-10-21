"""
2026-10-10: c2 (max_cardinality=2), MIXEDCLASS - Baseline vs Request-only vs Pair-only vs Combined,
mean over seeds (bars), sample std (error bars, ddof=1), individual seeds as dots, n annotated.
All thresholds 0.5 (Request Pruner expert-prefix-label MLP, min_retention 0.3; Pair Pruner GNNv1).
HPs: epochs=5, lr=0.0001, legacy scoring, pickup_slack=False; mixedclass = 39 train / 12 val files.

Per-seed "Best val service rate" (percent), seeds 1,2,3,4,42 unless noted:
  Baseline: sil_training_{cfg}_mixedclass_requestpruner_baseline_seed{s}.log
  Request-only: sil_training_{cfg}_mixedclass_requestpruner_prefixlabel_t0.5_seed{s}.log
                (bi400/ss80: seed 1 = 37.40% added 2026-10-10 from the cluster log -> n=5)
  Pair-only: sil_training_{cfg}_mixedclass_pairprunerv1final_pairt0.5_seed{s}.log
  Combined: sil_training_{cfg}_mixedclass_combined_reqprefix_pairv1final_pairt0.5_seed{s}.log
            (only seed 42 for bi200/ss100 and bi400/ss200 -> n=1, no error bar)

Run: ./venv/bin/python3 scripts/analysis/make_c2_mixedclass_variants_5seed_chart.py
"""
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
VARIANTS = ["Baseline (no pruner)", "Request Pruner only (t=0.5)", "Pair Pruner only (t=0.5)",
            "Combined (Request t=0.5, Pair t=0.5)"]
COLORS = ["#1E2761", "#276575", "#E8A33D", "#3E8F6C"]

SEEDS = {
    CONFIGS[0]: [[67.58, 66.29, 67.09, 62.60, 60.67], [33.71, 34.51, 31.46, 31.78, 35.31],
                 [65.49, 65.81, 64.04, 63.08, 60.03], [33.39]],
    CONFIGS[1]: [[47.51, 47.99, 45.10, 46.71, 43.50], [49.28, 48.64, 44.78, 51.85, 49.44],
                 [46.55, 43.82, 42.54, 46.71, 41.89], [47.67]],
    CONFIGS[2]: [[57.14, 54.74, 48.96, 52.65, 49.44], [37.40, 35.15, 31.14, 36.76, 36.44],
                 [60.19, 56.34, 52.01, 53.61, 49.12], [39.49, 33.87, 30.50, 35.79, 37.72]],
}


def stats(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else None)


def chart():
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    x = np.arange(len(CONFIGS))
    n, w = len(VARIANTS), 0.2
    rng = np.random.default_rng(0)
    for i, (var, color) in enumerate(zip(VARIANTS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            xs = SEEDS[cfg][i]
            m, s = stats(xs)
            ax.bar(xi + offs, m, width=w, color=color, zorder=3, label=var,
                   yerr=s if s is not None else None, capsize=4, ecolor="#222222")
            ax.scatter(xi + offs + rng.uniform(-0.04, 0.04, len(xs)), xs, s=14, color="white",
                       edgecolor="#222222", linewidth=0.7, zorder=5)
            top = m + (s or 0)
            lab = f"{m:.1f}" + (f"\u00b1{s:.1f}" if s is not None else "") + f"\n(n={len(xs)})"
            ax.annotate(lab, (xi + offs, top), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10.5)
    ax.set_ylabel("Best val service rate (%)", fontsize=11)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("c2 mixedclass: Baseline vs. Request Pruner vs. Pair Pruner vs. Combined (mean \u00b1 std over seeds)",
                 fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=9, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    fig.text(0.5, 0.005, "c2, mixedclass (39 train / 12 val files), epochs=5, lr=0.0001, legacy scoring, pickup_slack off; "
             "dots = single seeds, bars = mean, error bars = sample std; n=1 has no error bar",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / "c2_mixedclass_variants_5seed_service_rate.png", dpi=200)
    fig.savefig(OUT_DIR / "c2_mixedclass_variants_5seed_service_rate.pdf")
    plt.close(fig)
    print("saved c2_mixedclass_variants_5seed_service_rate.png/.pdf")


def table():
    heads = ["Baseline", "Request only", "Pair only", "Combined"]
    rows = []
    for cfg in CONFIGS:
        cells = []
        for xs in SEEDS[cfg]:
            m, s = stats(xs)
            cells.append(f"{m:.2f}" + (f" $\\pm$ {s:.2f}" if s is not None else "") + f" ($n{{=}}{len(xs)}$)")
        rows.append(f"{cfg.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    tex = (
        "\\begin{table}[h]\n\\centering\n\\small\n"
        "\\caption{c2 mixedclass: best val service rate (\\%), mean $\\pm$ sample std over seeds. "
        "Request Pruner $t{=}0.5$ (min retention 0.3), Pair Pruner GNNv1 $t{=}0.5$; epochs 5, lr 0.0001.}\n"
        "\\begin{tabular}{l" + "c" * len(heads) + "}\n\\toprule\n"
        "Config & " + " & ".join(heads) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_DIR / "c2_mixedclass_variants_5seed_table.tex").write_text(tex)
    print("saved c2_mixedclass_variants_5seed_table.tex")


if __name__ == "__main__":
    chart()
    table()
