"""
2026-10-09: c2 (max_cardinality=2), MIXEDCLASS, seed=42 - Baseline vs Request-only vs Pair-only vs
Combined (all thresholds 0.5), grouped bar charts (Service Rate, Runtime) + LaTeX table.
Same seed everywhere so cells are directly comparable (single seed, no error bars).
HPs: epochs=5, lr=0.0001, legacy scoring, pickup_slack=False (c2 convention), Request Pruner
expert-prefix-label MLP per config (t=0.5, min_retention 0.3), Pair Pruner GNNv1
(rgnn_mixed_c2_pw2_v1_final, t=0.5); mixedclass = class1 split + CLASS2_EXTRA (39 train / 12 val).

Sources ("Best val service rate", all seed 42):
  Baseline: sil_training_{cfg}_mixedclass_requestpruner_baseline_seed42.log
            (bi400/ss80 seed42 baseline added 2026-10-13: 49.44 %, sacct Elapsed 6:51:19, job 5653076 c2_mc_base_bi400ss80_s42)
  Request-only: sil_training_{cfg}_mixedclass_requestpruner_prefixlabel_t0.5_seed42.log
  Pair-only: sil_training_{cfg}_mixedclass_pairprunerv1final_pairt0.5_seed42.log (jobs 5650560-62)
  Combined: sil_training_{cfg}_mixedclass_combined_reqprefix_pairv1final_pairt0.5_seed42.log
Runtime: first-to-last log timestamp (Baseline, Request-only, Combined); sacct Elapsed for Pair-only
(jobs 5650560-62). Both include the post-training check over the training files.

Run: ./venv/bin/python3 scripts/analysis/make_c2_mixedclass_variants_chart.py
"""
# 2026-10-14: train count corrected 38 -> 39 files (23 class-1 + 16 CLASS2_EXTRA, verified in training_loop.py and the
# run logs "Train: 39 files, Val: 12 files"); earlier footers said 38.
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

SERVICE = {
    CONFIGS[0]: [60.67, 35.31, 60.03, 33.39],
    CONFIGS[1]: [43.50, 49.44, 41.89, 47.67],
    CONFIGS[2]: [49.44, 36.44, 49.12, 37.72],  # 2026-10-13: baseline bi400/ss80 seed 42 = 49.44 (fill-in job 5653076)
}
RUNTIME_H = {
    CONFIGS[0]: [4.65, 1.80, 2.99, 1.33],
    CONFIGS[1]: [3.06, 1.68, 1.81, 1.03],
    CONFIGS[2]: [6.86, 1.94, 3.81, 1.78],  # 2026-10-13: baseline bi400/ss80 seed 42 runtime = 6:51:19 (sacct, job 5653076)
}


def grouped_bars(data, ylabel, title, stem, fmt, ylim, pct=False):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(CONFIGS))
    n, w = len(VARIANTS), 0.2
    for i, (var, color) in enumerate(zip(VARIANTS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            v = data[cfg][i]
            if v is None:
                ax.annotate("n/a", (xi + offs, 0.5), ha="center", fontsize=9, color="#888888")
                continue
            ax.bar(xi + offs, v, width=w, color=color, zorder=3, label=var)
            ax.annotate(fmt.format(v), (xi + offs, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, fontweight="bold")
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
              fontsize=9, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "c2, mixedclass (39 train / 12 val files), seed=42, epochs=5, lr=0.0001, legacy scoring, "
             "pickup_slack off; Request Pruner min_retention 0.3; Pair Pruner GNNv1", ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"saved {OUT_DIR / stem}.png/.pdf")


def write_table():
    heads = ["Baseline", "Request only", "Pair only", "Combined"]
    rows = []
    for cfg in CONFIGS:
        cells = []
        for s, r in zip(SERVICE[cfg], RUNTIME_H[cfg]):
            cells.append("n/a" if s is None else f"{s:.2f}\\% / {r:.2f}\\,h")
        rows.append(f"{cfg.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    tex = (
        "\\begin{table}[h]\n\\centering\n\\small\n"
        "\\caption{c2 mixedclass, seed 42: best val service rate / runtime. Request Pruner $t{=}0.5$ "
        "(min retention 0.3), Pair Pruner GNNv1 $t{=}0.5$; epochs 5, lr 0.0001. "
        "Runtime includes the post-training check.}\n"
        "\\begin{tabular}{l" + "c" * len(heads) + "}\n\\toprule\n"
        "Config & " + " & ".join(heads) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_DIR / "c2_mixedclass_variants_table.tex").write_text(tex)
    print(f"saved {OUT_DIR / 'c2_mixedclass_variants_table.tex'}")


if __name__ == "__main__":
    grouped_bars(SERVICE, "Best val service rate (%)",
                 "c2 mixedclass: Baseline vs. Request Pruner vs. Pair Pruner vs. Combined (Service Rate)",
                 "c2_mixedclass_variants_service_rate", "{:.1f}", (0, 80), pct=True)
    grouped_bars(RUNTIME_H, "Runtime (hours, 5-epoch training run)",
                 "c2 mixedclass: Baseline vs. Request Pruner vs. Pair Pruner vs. Combined (Runtime)",
                 "c2_mixedclass_variants_runtime", "{:.1f}", (0, 5.5))
    write_table()
