"""
2026-10-19: c3 MIXEDCLASS-LITE, batch_interval=400/step_size=80 - RUNTIME (bar chart) and service rate vs. runtime (trade-off),
seed 42, epochs 5, learning_rate 0.0001, lite split (34 train / 12 val files). Prepared while runs are still in progress:
rerun after editing the DATA table below.
Runtime = sacct Elapsed (hours, incl. the post-training check over the training files), jobs 5670939-5670948 (2026-10-19):
  pair-only 1-02:50:31, Request-only t=0.6 19:16:09 / t=0.7 18:20:32, Combined t=0.6 05:47:03 / t=0.7 06:10:35 / t=0.8 05:22:54 /
  t=0.9 03:00:27.  Request-only t=0.8 15:19:39, t=0.9 07:40:53 (COMPLETED, sacct 2026-10-19).  Baseline: ESTIMATE ~93 h = 5 x 18.7 h (epoch 1 measured, job 5670939 still running) - hatched bar,
  replace with the sacct Elapsed once finished (set BASELINE_FINAL = True).
Best val service rate (%): from the "Best val service rate" log lines (pasted into the chat on 2026-10-19); baseline: epoch 1 only
(76.24 %), not used in the trade-off panel until final.
Run: ./venv/bin/python3 scripts/analysis/make_c3_mixedclass_lite_runtime_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
BASELINE_FINAL = False
BASELINE_RUNTIME_H = 93.5      # estimate until the job finishes
BASELINE_BEST_VAL = None       # fill in when finished

# label, runtime h (None = running), best val % (None = running), family
DATA = [
    ("Baseline", BASELINE_RUNTIME_H, BASELINE_BEST_VAL, "base"),
    ("Pair only", 26.84, 78.49, "pair"),
    ("Request t=0.6", 19.27, 45.75, "req"),
    ("Request t=0.7", 18.34, 44.14, "req"),
    ("Request t=0.8", 15.33, 34.83, "req"),   # 2026-10-19: sacct 15:19:39, COMPLETED
    ("Request t=0.9", 7.68, 31.46, "req"),    # 2026-10-19: sacct 07:40:53, best val 31.46 % at epoch 1 (epoch 5: 27.61 %)
    ("Combined t=0.6", 5.78, 41.73, "comb"),
    ("Combined t=0.7", 6.18, 42.70, "comb"),
    ("Combined t=0.8", 5.38, 34.19, "comb"),
    ("Combined t=0.9", 3.01, 31.62, "comb"),
]
COLOR = {"base": "#1E2761", "pair": "#E8A33D", "req": "#276575", "comb": "#3E8F6C"}


def bars():
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for i, (lab, rt, bv, fam) in enumerate(DATA):
        if rt is None:
            ax.annotate("running", (i, 0.5), ha="center", fontsize=9, color="#888888")
            continue
        est = fam == "base" and not BASELINE_FINAL
        ax.bar(i, rt, color=COLOR[fam], width=0.7, zorder=3, hatch="//" if est else None, edgecolor="white" if est else None,
               alpha=0.75 if est else 1.0)
        ax.annotate(f"{rt:.1f}" + (" (est.)" if est else ""), (i, rt), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(DATA)))
    ax.set_xticklabels([d[0] for d in DATA], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Runtime (hours, incl. post-training check)", fontsize=10.5)
    ax.set_ylim(0, 105)
    ax.set_title("c3 mixedclass-lite, batch_interval=400/step_size=80: runtime per variant (seed 42, 5 epochs)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "Baseline hatched = estimate (5 x 18.7 h; job still running). Lite split: 34 train files, "
             "12 val files; epochs=5, lr=0.0001, legacy scoring.", ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_lite_bi400ss80_runtime.{ext}", dpi=200)
    plt.close(fig)


def tradeoff():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for lab, rt, bv, fam in DATA:
        if rt is None or bv is None:
            continue
        marker = "*" if fam == "base" else "o"
        ax.scatter([rt], [bv], s=260 if fam == "base" else 90, marker=marker, color=COLOR[fam], edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.annotate(lab.replace("Request ", "Req ").replace("Combined ", "Comb "), (rt, bv), textcoords="offset points",
                    xytext=(7, 5), fontsize=8.5, color="#222222")
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (hours, log scale)", fontsize=10.5)
    ax.set_ylabel("Best val service rate (%)", fontsize=10.5)
    ax.set_ylim(20, 90)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("c3 mixedclass-lite, batch_interval=400/step_size=80: service rate vs. runtime", fontsize=11.5)
    ax.grid(True, linestyle=":", alpha=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "Baseline not shown until finished. Seed 42, 5 epochs, lite split.", ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_lite_bi400ss80_tradeoff.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    bars()
    tradeoff()
    print("saved c3_mixedclass_lite_bi400ss80_runtime / _tradeoff (.png/.pdf)")
