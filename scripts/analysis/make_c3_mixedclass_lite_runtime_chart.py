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
# 2026-10-22: baseline job 5670939 finished: sacct Elapsed 4-17:23:51 = 113.40 h (incl. post-training check), best val 80.26 % at epoch 4 (epoch 5: 80.10 %), log of 2026-10-22
BASELINE_FINAL = True
BASELINE_RUNTIME_H = 113.40
BASELINE_BEST_VAL = 80.26

# label, runtime h (None = running), best val % (None = running), family
DATA = [
    ("Baseline", BASELINE_RUNTIME_H, BASELINE_BEST_VAL, "base"),
    ("Pair only", 26.84, 78.49, "pair"),
    # 2026-10-22: Combined t=0.1-0.4 (Pair 0.5): sacct Elapsed 1-00:11:37 / 1-23:20:13 / 1-22:11:08 / 2-04:46:07, best val from the "Best val service rate" log lines (epochs 5 / 3 / 4 / 5)
    ("Combined t=0.1", 24.19, 78.49, "comb"),
    ("Combined t=0.2", 47.34, 78.81, "comb"),
    ("Combined t=0.3", 46.19, 78.81, "comb"),
    ("Combined t=0.4", 52.77, 79.29, "comb"),
    # 2026-10-26: Request-only t=0.1 / t=0.2 finished: sacct Elapsed 4-09:12:59 / 4-12:52:53, best val from the "Best val service rate" log lines (epoch 5 / epoch 4); t=0.3 / 0.4 still running (epoch 4: 79.94 / 80.10 %)
    ("Request t=0.1", 105.22, 80.42, "req"),
    ("Request t=0.2", 108.88, 80.74, "req"),
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
    ax.set_ylim(0, 125)
    ax.set_title("c3 mixedclass-lite, batch_interval=400/step_size=80: runtime per variant (seed 42, 5 epochs)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "Runtime = sacct Elapsed incl. post-training check, node load varied. Lite split: 34 train files, "
             "12 val files; epochs=5, lr=0.0001, legacy scoring.", ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_lite_bi400ss80_runtime.{ext}", dpi=200)
    plt.close(fig)


# 2026-10-22: per-label offsets (points) for clustered markers, default (7, 5)
OFFS = {"Combined t=0.1": (-70, 6), "Pair only": (7, -15), "Combined t=0.2": (-48, -17), "Combined t=0.3": (-34, 9), "Combined t=0.4": (8, -15),
        "Combined t=0.6": (-66, -12), "Combined t=0.7": (7, 5), "Request t=0.6": (7, 5), "Request t=0.7": (-62, -4),
        "Request t=0.1": (-12, 12), "Request t=0.2": (-62, 8)}


def tradeoff():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for lab, rt, bv, fam in DATA:
        if rt is None or bv is None:
            continue
        marker = "*" if fam == "base" else "o"
        ax.scatter([rt], [bv], s=260 if fam == "base" else 90, marker=marker, color=COLOR[fam], edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.annotate(lab.replace("Request ", "Req ").replace("Combined ", "Comb "), (rt, bv), textcoords="offset points",
                    xytext=OFFS.get(lab, (7, 5)), fontsize=8.5, color="#222222")
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (hours, log scale)", fontsize=10.5)
    ax.set_ylabel("Best val service rate (%)", fontsize=10.5)
    ax.set_ylim(20, 90)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("c3 mixedclass-lite, batch_interval=400/step_size=80: service rate vs. runtime", fontsize=11.5)
    ax.grid(True, linestyle=":", alpha=0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "Seed 42, 5 epochs, lite split; runtime = sacct Elapsed incl. post-training check (node load varied). Request-only t=0.3 / 0.4 still running.", ha="center", fontsize=7.8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_lite_bi400ss80_tradeoff.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    bars()
    tradeoff()
    print("saved c3_mixedclass_lite_bi400ss80_runtime / _tradeoff (.png/.pdf)")
