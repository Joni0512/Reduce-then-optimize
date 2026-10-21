"""
2026-10-13: c2 MIXEDCLASS, Request Pruner only (expert-prefix-label MLP, min_retention 0.3): RUNTIME per threshold
t = 0.1 ... 0.8 vs. the no-pruner baseline - runtime counterpart of make_c2_mixedclass_request_threshold_bars.py.
Bar = mean of the available seeds (42 and 1), dots = single seeds. Runtime includes the post-training check
over the training files (as in the other runtime charts).

Sources:
  sacct Elapsed (cluster, 2026-10-13) of jobs c2_mc_reqt01..08_{cfg}_s{42,1} (seeds 42/1; t=0.3/0.4 and t=0.2
  bi400/ss80 from the completed jobs, the CANCELLED duplicates were skipped) and c2_mc_base_bi400ss80_s{42,1}.
  Single seed 42 only, from the first-to-last log timestamp (make_c2_mixedclass_variants_chart.py, 2026-10-09):
  baseline bi200/ss100 4.65 h, baseline bi400/ss200 3.06 h, and t=0.5 (1.80 / 1.68 / 1.94 h).
  2026-10-13: bi200/ss100 t=0.3 / t=0.4 (no sacct entry, older job names) from the first-to-last timestamp of the
  complete local logs sil_training_bi200_ss100_mixedclass_requestpruner_prefixlabel_t0.{3,4}_seed{42,1}.log
  (t0.3: 2:19:50 / 2:25:56, t0.4: 1:54:24 / 2:00:50; both end with 'Run complete').
HPs: epochs 5, lr 0.0001, legacy scoring, pickup_slack off.

Run: ./venv/bin/python3 scripts/analysis/make_c2_mixedclass_request_threshold_runtime.py
"""
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["batch_interval=200/step_size=100", "batch_interval=400/step_size=200", "batch_interval=400/step_size=80"]
BARS = ["Baseline", "t=0.1", "t=0.2", "t=0.3", "t=0.4", "t=0.5", "t=0.6", "t=0.7", "t=0.8"]
COLORS = ["#1E2761", "#D9EEE5", "#BFE3D3", "#A9D6C4", "#7FBF9F", "#3E8F6C", "#34795A", "#2A6B4D", "#14402D"]

# per bar: list of runtimes ("h:mm:ss" from sacct, or hours as float from logs), seeds [42, 1]; None = n/a
DATA = {
    CONFIGS[0]: [[4.65], ["3:25:45", "3:18:15"], ["2:30:10", "2:38:13"], ["2:19:50", "2:25:56"],
                 ["1:54:24", "2:00:50"], [1.80],
                 ["1:54:45", "1:40:09"], ["1:23:44", "1:36:54"], ["1:26:16", "1:26:39"]],
    CONFIGS[1]: [[3.06], ["1:51:32", "2:00:23"], ["1:40:16", "1:38:02"], ["1:36:14", "1:43:24"],
                 ["1:36:04", "1:33:47"], [1.68], ["1:34:37", "1:34:27"], ["1:30:12", "1:34:52"],
                 ["1:24:54", "1:25:55"]],
    CONFIGS[2]: [["6:51:19", "6:20:55"], ["3:19:33", "3:21:18"], ["3:21:15", "2:55:41"], ["2:34:07", "2:36:34"],
                 ["2:26:36", "2:16:23"], [1.94], ["1:52:05", "2:15:25"], ["1:42:30", "1:48:50"],
                 ["1:35:44", "1:53:06"]],
}


def hours(v):
    if isinstance(v, str):
        h, m, s = (int(x) for x in v.split(":"))
        return h + m / 60 + s / 3600
    return float(v)


def main():
    fig, ax = plt.subplots(figsize=(16, 6.8))
    x = np.arange(len(CONFIGS))
    n, w = len(BARS), 0.1
    for i, (name, color) in enumerate(zip(BARS, COLORS)):
        offs = (i - (n - 1) / 2) * w
        for xi, cfg in zip(x, CONFIGS):
            d = DATA[cfg][i]
            if d is None:
                ax.annotate("n/a", (xi + offs, 0.15), ha="center", fontsize=8, color="#888888")
                continue
            vals = [hours(v) for v in d]
            m = st.mean(vals)
            ax.bar(xi + offs, m, width=w, color=color, zorder=3, label=name)
            ax.scatter([xi + offs] * len(vals), vals, s=16, color="white", edgecolor="#222222", linewidth=0.7, zorder=5)
            ax.annotate(f"{m:.1f}", (xi + offs, max(vals)), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10.5)
    ax.set_ylabel("Runtime (hours, incl. post-training check)", fontsize=11)
    ax.set_ylim(0, 8)
    ax.set_title("c2 mixedclass, Request Pruner only: runtime per threshold (t = 0.1 ... 0.8) vs. baseline", fontsize=12)
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False, ncol=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "c2, mixedclass, bar = mean, dots = single seeds (42, 1); one dot = seed 42 only (baseline bi200/ss100 "
             "and bi400/ss200, t=0.5); t=0.3/0.4 at bi200/ss100 from log timestamps; epochs=5, lr=0.0001, pickup_slack off, min_retention 0.3",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c2_mixedclass_request_threshold_runtime.{ext}", dpi=200)
    print("saved c2_mixedclass_request_threshold_runtime.png/.pdf")


if __name__ == "__main__":
    main()
