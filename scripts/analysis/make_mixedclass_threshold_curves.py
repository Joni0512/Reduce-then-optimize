"""
2026-10-10: MIXEDCLASS threshold curves (pruning rate vs. service rate), two figures + tables:

(1) c2, Request-only (expert-prefix-label MLP, min_retention 0.3), t in {0.3, 0.5, 0.7, 0.8}:
    service rate = mean over seeds 42 and 1 (same seeds for every threshold and for the baseline),
    pruning rate = mean of the logs ("Request pruner: kept K/N ... M pruned" via compute_pruning_rates.py).
    Per-seed "Best val service rate" (percent), seeds 42 / 1:
      bi200/ss100: t0.3 45.26/56.50  t0.5 35.31/33.71  t0.7 33.39/34.19  t0.8 30.98/34.51  baseline 60.67/67.58
      bi400/ss200: t0.3 50.72/47.35  t0.5 49.44/49.28  t0.7 48.48/44.94  t0.8 45.10/43.66  baseline 43.50/47.51
      bi400/ss80 : t0.3 32.42/35.15  t0.5 36.44/37.40  t0.7 33.39/31.62  t0.8 30.98/30.82  baseline 49.44/57.14
    Pruning rate (percent, mean of the available logs): see dict below.
(2) c3, Combined (Pair Pruner t=0.5 fixed), Request t in {0.5, 0.7, 0.8, 0.9}, seed 42 only:
    sources: sil_training_{cfg}_c3_mixedclass_combined_reqprefix_{reqt{t}_}pairv1final_pairt0.5_seed42.log;
    baseline only for bi400/ss200 (62.76%, sil_training_bi400_ss200_c3_mixedclass_legacy_seed42.log).
HPs: epochs 5, lr 0.0001, legacy scoring; c2 pickup_slack off, c3 pickup_slack on (class default).

Run: ./venv/bin/python3 scripts/analysis/make_mixedclass_threshold_curves.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

C2_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
C3_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
C2_DIR.mkdir(parents=True, exist_ok=True)
C3_DIR.mkdir(parents=True, exist_ok=True)

LABEL = {"bi200_ss100": "batch_interval=200/step_size=100",
         "bi400_ss200": "batch_interval=400/step_size=200",
         "bi400_ss80": "batch_interval=400/step_size=80"}
COLOR = {"bi200_ss100": "#276575", "bi400_ss200": "#A6493A", "bi400_ss80": "#3E8F6C"}

# (threshold, pruning rate %, service rate % (mean of seeds 42 and 1)); 2026-10-13: t=0.6 added (logs of 2026-10-12)
C2 = {
    "bi200_ss100": [(0.1, 11.4, 56.82), (0.2, 24.4, 54.66), (0.3, 29.9, 50.88), (0.4, 34.8, 39.81),
                    (0.5, 39.1, 34.51), (0.6, 42.1, 34.91), (0.7, 45.3, 33.79), (0.8, 47.9, 32.75)],
    "bi400_ss200": [(0.1, 22.7, 48.08), (0.2, 28.9, 49.20), (0.3, 32.1, 49.04), (0.4, 34.0, 49.20),
                    (0.5, 35.5, 49.36), (0.6, 36.9, 48.88), (0.7, 38.3, 46.71), (0.8, 39.1, 44.38)],
    "bi400_ss80": [(0.1, 23.6, 47.11), (0.2, 31.7, 50.16), (0.3, 35.8, 33.79), (0.4, 39.3, 40.77),
                   (0.5, 42.8, 36.92), (0.6, 46.0, 32.59), (0.7, 48.6, 32.51), (0.8, 50.9, 30.90)],
}
C2_BASELINE = {"bi200_ss100": 64.13, "bi400_ss200": 45.51, "bi400_ss80": 53.29}

C3 = {
    "bi200_ss100": [(0.7, 25.81, 31.30), (0.8, 31.32, 30.98), (0.9, 39.65, 24.72)],
    "bi400_ss200": [(0.5, 1.67, 61.80), (0.7, 28.94, 60.51), (0.8, 32.19, 61.16), (0.9, 35.69, 53.45)],
    "bi400_ss80": [(0.7, 31.47, 52.17), (0.8, 36.63, 39.65), (0.9, 42.94, 36.60)],
}
C3_BASELINE = {"bi400_ss200": 62.76}


def curve(data, baseline, title, footer, out_dir, stem, baseline_label_right=False):
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    for cfg, pts in data.items():
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, marker="o", color=COLOR[cfg], linewidth=2, markersize=7, label=LABEL[cfg], zorder=3)
        for t, x, y in [(p[0], p[1], p[2]) for p in pts]:
            ax.annotate(f"t={t}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=COLOR[cfg])
        if cfg in baseline:
            ax.axhline(baseline[cfg], color=COLOR[cfg], linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
            # 2026-10-12: c3 curve - label at the left end collided with the "t=0.5" point label (point at
            # ~1.7 % pruning sits right under the baseline). baseline_label_right moves it to the right axis edge.
            if baseline_label_right:
                ax.annotate(f"baseline {baseline[cfg]:.1f}%", (1.0, baseline[cfg]),
                            xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(-4, 4),
                            ha="right", fontsize=8, color=COLOR[cfg])
            else:
                ax.annotate(f"baseline {baseline[cfg]:.1f}%", (min(xs) - 0.5, baseline[cfg]),
                            textcoords="offset points", xytext=(0, 4), fontsize=8, color=COLOR[cfg])
    ax.set_xlabel("Request Pruner: pruned share of requests (%)", fontsize=11)
    ax.set_ylabel("Best val service rate (%)", fontsize=11)
    ax.set_ylim(15, 75)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left", fontsize=9.5, frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"saved {out_dir / stem}.png/.pdf")


def table(data, baseline, caption, out_dir, stem, thresholds):
    heads = ["Baseline"] + [f"$t{{=}}{t}$" for t in thresholds]
    rows = []
    for cfg, pts in data.items():
        d = {p[0]: p for p in pts}
        cells = [f"{baseline[cfg]:.1f}\\%" if cfg in baseline else "n/a"]
        for t in thresholds:
            cells.append(f"{d[t][2]:.1f}\\% ({d[t][1]:.1f}\\%)" if t in d else "n/a")
        rows.append(f"{LABEL[cfg].replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    tex = ("\\begin{table}[h]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
           "\\begin{tabular}{l" + "c" * len(heads) + "}\n\\toprule\nConfig & " + " & ".join(heads)
           + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (out_dir / f"{stem}_table.tex").write_text(tex)
    print(f"saved {out_dir / stem}_table.tex")


if __name__ == "__main__":
    curve(C2, C2_BASELINE,
          "c2 mixedclass, Request Pruner only: pruning rate vs. service rate (t = 0.1 ... 0.8, incl. 0.6)",
          "c2, mixedclass, mean of seeds 42 and 1 (baseline on the same seeds), epochs=5, lr=0.0001, pickup_slack off, min_retention 0.3",
          C2_DIR, "c2_mixedclass_request_threshold_curve")
    table(C2, C2_BASELINE,
          "c2 mixedclass, Request Pruner only: service rate (pruned share of requests) per threshold, mean of seeds 42 and 1; "
          "baseline on the same seeds; epochs 5, lr 0.0001, min retention 0.3.",
          C2_DIR, "c2_mixedclass_request_threshold", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    curve(C3, C3_BASELINE,
          "c3 mixedclass, Combined (Pair Pruner t=0.5): pruning rate vs. service rate (Request t = 0.5 ... 0.9)",
          "c3, mixedclass, seed 42 only, epochs=5, lr=0.0001, pickup_slack on; baseline available for batch_interval=400/step_size=200 only",
          C3_DIR, "c3_mixedclass_combined_threshold_curve", baseline_label_right=True)
    table(C3, C3_BASELINE,
          "c3 mixedclass, Combined (Pair Pruner $t{=}0.5$): service rate (Request-Pruner pruned share) per Request threshold, "
          "seed 42; epochs 5, lr 0.0001; n/a = not run/cancelled; baseline only for one config.",
          C3_DIR, "c3_mixedclass_combined_threshold", [0.5, 0.7, 0.8, 0.9])
