"""
2026-10-04: c3 (max_cardinality=3), class1, seed=42 - Baseline vs Request-only vs Pair-only vs
Combined (Request Pruner threshold swept 0.5/0.7/0.8/0.9, Pair Pruner fixed 0.5), grouped bar
charts (Service Rate, Runtime) + LaTeX table. Request Pruner: min_retention_fraction=0.3,
expert-prefix-label MLP per config; Pair Pruner GNNv1 (rgnn_mixed_c2_pw2_v1_final); epochs=5,
lr=0.0001, legacy scoring, pickup_slack=True (class default, c3 internally consistent).

Sources ("Best val service rate" lines, chat 2026-09-24..10-04):
  Baseline: sil_training_bi200_ss100_c3_class1_legacy.log (seed42),
            sil_training_bi400_ss200_c3_class1_legacy_seed42.log,
            sil_training_bi400_ss80_c3_class1_legacy_seed42.log (added 2026-10-13, 88.99 %, 10.18 h)
  Request-only: sil_training_{cfg}_c3_class1_requestpruner_prefixlabel_t0.5_seed42.log
  Pair-only: sil_training_{cfg}_c3_class1_pairprunerv1final_pairt0.5_seed42.log
  Combined t=0.5: sil_training_{cfg}_c3_class1_combined_reqprefix_pairv1final_pairt0.5_seed42.log
  Combined t=0.7/0.8/0.9: sil_training_{cfg}_c3_class1_combined_reqprefix_reqt{t}_pairv1final_pairt0.5_seed42.log
Runtime: log first-to-last timestamp (all except Pair-only); sacct Elapsed for Pair-only
(jobs 5636302-5636304).

Run: ./venv/bin/python3 scripts/analysis/make_c3_class1_four_variants_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
# 2026-10-04: thresholds in the labels; Combined swept over the Request Pruner threshold (Pair fixed 0.5)
VARIANTS = ["Baseline (no pruner)", "Request Pruner only (t=0.5)", "Pair Pruner only (t=0.5)",
            "Combined (Request t=0.5, Pair t=0.5)", "Combined (Request t=0.7, Pair t=0.5)",
            "Combined (Request t=0.8, Pair t=0.5)", "Combined (Request t=0.9, Pair t=0.5)"]
COLORS = ["#1E2761", "#276575", "#E8A33D", "#7FBF9F", "#3E8F6C", "#2A6B4D", "#14402D"]

SERVICE = {  # None = missing
    CONFIGS[0]: [86.79, 86.48, 87.11, 86.16, 87.11, 72.64, 46.86],
    CONFIGS[1]: [68.55, 70.13, 66.98, 68.55, 69.81, 69.81, 66.04],
    # 2026-10-13: baseline bi400/ss80 = 88.99 % (Best val at epoch 5; epochs 1-5: 84.28/87.74/88.68/87.74/88.99), from the
    # cluster log sil_training_bi400_ss80_c3_class1_legacy_seed42.log (run 2026-09-26 19:03 -> 2026-09-27 05:14)
    CONFIGS[2]: [88.99, 88.68, 87.74, 87.74, 88.99, 88.99, 66.35],
}
RUNTIME_H = {
    CONFIGS[0]: [8.50, 8.72, 2.36, 2.21, 2.24, 1.46, 0.82],
    CONFIGS[1]: [7.96, 7.72, 1.95, 2.22, 1.79, 1.75, 1.65],
    CONFIGS[2]: [10.18, 9.56, 2.54, 2.47, 2.49, 1.70, 1.06],  # 2026-10-13: baseline = log 19:03:18 -> 05:14:05
}


def grouped_bars(data, ylabel, title, stem, fmt, ylim, pct=False):
    fig, ax = plt.subplots(figsize=(15, 6.6))
    x = np.arange(len(CONFIGS))
    n = len(VARIANTS)
    w = 0.115
    for i, (var, color) in enumerate(zip(VARIANTS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            v = data[cfg][i]
            if v is None:
                ax.annotate("n/a", (xi + offs, 1), ha="center", fontsize=8, color="#888888")
                continue
            ax.bar(xi + offs, v, width=w, color=color, zorder=3, label=var if xi == 0 else None)
            ax.annotate(fmt.format(v), (xi + offs, v), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title(title, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=8.5, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "c3, class1, seed=42, epochs=5, lr=0.0001, legacy scoring; Request Pruner min_retention 0.3; "
             "Pair Pruner GNNv1",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"saved {OUT_DIR / stem}.png/.pdf")


def write_table():
    heads = ["Baseline", "Req.\\ only", "Pair only", "Comb.\\ $t_R{=}0.5$", "Comb.\\ $t_R{=}0.7$",
             "Comb.\\ $t_R{=}0.8$", "Comb.\\ $t_R{=}0.9$"]
    rows = []
    for cfg in CONFIGS:
        cells = []
        for s, r in zip(SERVICE[cfg], RUNTIME_H[cfg]):
            cells.append("n/a" if s is None else f"{s:.2f}\\% / {r:.2f}\\,h")
        rows.append(f"{cfg.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    tex = (
        "\\begin{table}[h]\n\\centering\n\\small\n"
        "\\caption{c3 class1, seed 42: best val service rate / runtime. $t_R$ = Request Pruner threshold "
        "(min retention 0.3); Pair Pruner GNNv1 threshold fixed at 0.5 (also for Pair only); "
        "Req.\\ only uses $t_R{=}0.5$; epochs 5, lr 0.0001.}\n"
        "\\begin{tabular}{l" + "c" * len(heads) + "}\n\\toprule\n"
        "Config & " + " & ".join(heads) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_DIR / "c3_class1_four_variants_table.tex").write_text(tex)
    print(f"saved {OUT_DIR / 'c3_class1_four_variants_table.tex'}")


if __name__ == "__main__":
    grouped_bars(SERVICE, "Best val service rate (%)",
                 "c3 class1: Baseline vs. Request Pruner vs. Pair Pruner vs. Combined (Service Rate)",
                 "c3_class1_four_variants_service_rate", "{:.1f}", (0, 105), pct=True)
    grouped_bars(RUNTIME_H, "Runtime (hours, 5-epoch training run)",
                 "c3 class1: Baseline vs. Request Pruner vs. Pair Pruner vs. Combined (Runtime)",
                 "c3_class1_four_variants_runtime", "{:.1f}", (0, 11.5))
    write_table()
