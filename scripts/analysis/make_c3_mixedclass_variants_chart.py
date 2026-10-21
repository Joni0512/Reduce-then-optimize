"""
2026-10-08: c3 (max_cardinality=3), MIXEDCLASS, seed=42 - Baseline vs Pair-only vs Combined
(Request Pruner threshold swept 0.5/0.7/0.8/0.9, Pair Pruner fixed 0.5), grouped bar charts
(Service Rate, Runtime) + LaTeX table. Request-only is missing for mixedclass (jobs cancelled).
Same HPs as the class1 chart: epochs=5, lr=0.0001, legacy scoring, Request Pruner min_retention 0.3
(expert-prefix-label MLP per config), Pair Pruner GNNv1 (rgnn_mixed_c2_pw2_v1_final),
pickup_slack=True (class default). Train/val split: class1 split + CLASS2_EXTRA (39 train / 12 val files).

Sources ("Best val service rate", chat 2026-09-27..10-08, all seed 42):
  Baseline bi400/ss200: sil_training_bi400_ss200_c3_mixedclass_legacy_seed42.log (62.76%, epoch 5;
      job FAILED only in the post-training train_val check with a Gurobi token-server error)
  Pair-only: ..._c3_mixedclass_pairprunerv1final_pairt0.5_seed42.log (bi200/ss100 68.06%, bi400/ss200 55.54%)
  Combined: ..._c3_mixedclass_combined_reqprefix_reqt{t}_pairv1final_pairt0.5_seed42.log
Runtime = sacct Elapsed of the job (includes the post-training train_val check over all training
files; node load varied). Baseline runtime omitted: its job crashed inside that check, so its Elapsed
is not a full-run time. Missing cells (n/a) = not run / cancelled / still running.

Run: ./venv/bin/python3 scripts/analysis/make_c3_mixedclass_variants_chart.py
"""
# 2026-10-14: train count corrected 38 -> 39 files (23 class-1 + 16 CLASS2_EXTRA, verified in training_loop.py and the
# run logs "Train: 39 files, Val: 12 files"); earlier footers said 38.
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
VARIANTS = ["Baseline (no pruner)", "Pair Pruner only (t=0.5)",
            "Combined (Request t=0.5, Pair t=0.5)", "Combined (Request t=0.7, Pair t=0.5)",
            "Combined (Request t=0.8, Pair t=0.5)", "Combined (Request t=0.9, Pair t=0.5)"]
COLORS = ["#1E2761", "#E8A33D", "#7FBF9F", "#3E8F6C", "#2A6B4D", "#14402D"]

# 2026-10-12: Combined t=0.5 batch_interval=200/step_size=100 = 45.26 %. Source: epoch lines of
# sil_training_bi200_ss100_c3_mixedclass_combined_reqprefix_reqt0.5_pairv1final_pairt0.5_seed42.log
# (epochs 1-5: 31.62 / 33.55 / 33.07 / 40.45 / 45.26 %, i.e. best val = epoch 5). The 24 h job (5633984) hit its
# time limit right after epoch 5 (inside the post-training check), so the result exists; its runtime is therefore
# not a full-run time and stays n/a in RUNTIME_H.
SERVICE = {
    CONFIGS[0]: [None, 68.06, 45.26, 31.30, 30.98, 24.72],
    CONFIGS[1]: [62.76, 55.54, 61.80, 60.51, 61.16, 53.45],
    CONFIGS[2]: [None, None, None, 52.17, 39.65, 36.60],
}
RUNTIME_H = {
    CONFIGS[0]: [None, 118.5, None, 22.9, 21.3, 13.9],
    CONFIGS[1]: [None, 12.6, 12.2, 7.0, 4.6, 4.1],
    CONFIGS[2]: [None, None, None, 28.7, 28.5, 22.5],
}


def grouped_bars(data, ylabel, title, stem, fmt, ylim, pct=False, clip=None):
    fig, ax = plt.subplots(figsize=(15, 6.6))
    x = np.arange(len(CONFIGS))
    n, w = len(VARIANTS), 0.13
    for i, (var, color) in enumerate(zip(VARIANTS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            v = data[cfg][i]
            if v is None:
                ax.annotate("n/a", (xi + offs, 0.5), ha="center", fontsize=8, color="#888888")
                continue
            # 2026-10-12: runtime chart only - the 118.5 h outlier (Pair-only batch_interval=200/step_size=100)
            # squashed all other bars. With clip set, bars above it are drawn up to `clip` with a white break
            # marker and the true value in the label; the data itself is unchanged.
            if clip is not None and v > clip:
                ax.bar(xi + offs, clip, width=w, color=color, zorder=3, label=var)
                ax.plot([xi + offs - w / 2, xi + offs + w / 2], [clip * 0.9, clip * 0.93], color="white",
                        linewidth=3, zorder=4)
                ax.annotate(fmt.format(v) + "\n(axis cut)", (xi + offs, clip), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7.5, fontweight="bold")
                continue
            ax.bar(xi + offs, v, width=w, color=color, zorder=3, label=var)
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
    fig.text(0.5, 0.005, "c3, mixedclass (39 train / 12 val files), seed=42, epochs=5, lr=0.0001, legacy scoring; "
             "Request Pruner min_retention 0.3; Pair Pruner GNNv1; n/a = not run / cancelled / still running",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"saved {OUT_DIR / stem}.png/.pdf")


def write_table():
    heads = ["Baseline", "Pair only", "Comb.\\ $t_R{=}0.5$", "Comb.\\ $t_R{=}0.7$",
             "Comb.\\ $t_R{=}0.8$", "Comb.\\ $t_R{=}0.9$"]
    rows = []
    for cfg in CONFIGS:
        cells = []
        for s, r in zip(SERVICE[cfg], RUNTIME_H[cfg]):
            if s is None:
                cells.append("n/a")
            elif r is None:
                cells.append(f"{s:.2f}\\% / n/a")
            else:
                cells.append(f"{s:.2f}\\% / {r:.1f}\\,h")
        rows.append(f"{cfg.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    tex = (
        "\\begin{table}[h]\n\\centering\n\\small\n"
        "\\caption{c3 mixedclass, seed 42: best val service rate / runtime. $t_R$ = Request Pruner threshold "
        "(min retention 0.3); Pair Pruner GNNv1 threshold fixed at 0.5; epochs 5, lr 0.0001. "
        "Runtime includes the post-training overfit check; n/a = not run, cancelled or still running.}\n"
        "\\begin{tabular}{l" + "c" * len(heads) + "}\n\\toprule\n"
        "Config & " + " & ".join(heads) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_DIR / "c3_mixedclass_variants_table.tex").write_text(tex)
    print(f"saved {OUT_DIR / 'c3_mixedclass_variants_table.tex'}")


if __name__ == "__main__":
    grouped_bars(SERVICE, "Best val service rate (%)",
                 "c3 mixedclass: Baseline vs. Pair Pruner vs. Combined (Service Rate)",
                 "c3_mixedclass_variants_service_rate", "{:.1f}", (0, 105), pct=True)
    grouped_bars(RUNTIME_H, "Runtime (hours, incl. post-training check)",
                 "c3 mixedclass: Baseline vs. Pair Pruner vs. Combined (Runtime)",
                 "c3_mixedclass_variants_runtime", "{:.1f}", (0, 40), clip=33)
    write_table()
