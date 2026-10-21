"""
2026-10-12: c2 mixedclass, Request Pruner only (t=0.5): pruned share of requests (top) vs. change in service rate
(bottom, Request-only minus Baseline) per class x geometry group, for the 12 test/val instances. Answers
"why do class 2 instances suffer more from pruning?" (see chat 2026-10-12).

Sources:
  damage:  figures_export/Request Pruner SIL/Cardinality 2/c2_mixedclass_per_instance_request_damage_table.csv
           (per-instance Request-only vs. Baseline, 5 seeds each, built by make_c2_per_instance_request_damage_chart.py);
           group value = mean of the group's instance deltas (2 instances per class x geometry group) in percentage points.
  pruning: sil_training_{cfg}_mixedclass_requestpruner_prefixlabel_t0.5_seed*.log in the repo root, pooled over seeds
           (bi200_ss100, bi400_ss200: seeds 1,2,3,4,42; bi400_ss80: seeds 2,3,4,42 - seed 1 log not local), via
           compute_pruning_rate_per_instance.py (pruned / requests seen over all windows and epochs of the log,
           includes the optimal_val pre-pass).
HPs: epochs=5, lr=0.0001, legacy scoring, pickup_slack off, Request Pruner expert-prefix-label MLP, min_retention 0.3.

Run: ./venv/bin/python3 scripts/analysis/make_c2_pruning_rate_vs_damage_chart.py
"""
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compute_pruning_rate_per_instance import TEST, parse  # noqa: E402

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
CSV_PATH = OUT_DIR / "c2_mixedclass_per_instance_request_damage_table.csv"
CFGS = [("bi200_ss100", "batch_interval=200/step_size=100"),
        ("bi400_ss200", "batch_interval=400/step_size=200"),
        ("bi400_ss80", "batch_interval=400/step_size=80")]
GROUPS = [("1", "lc"), ("1", "lr"), ("1", "lrc"), ("2", "lc"), ("2", "lr"), ("2", "lrc")]
CLS_COLOR = {"1": "#276575", "2": "#E8A33D"}


def group_of(stem):
    geo = re.match(r"[a-z]+", stem).group()
    return stem[len(geo)], geo


def pruning_rates(cfg):
    logs = sorted(glob.glob(f"sil_training_{cfg}_mixedclass_requestpruner_prefixlabel_t0.5_seed*.log"))
    seen, pruned = parse(logs)
    agg = defaultdict(lambda: [0, 0])
    for s in TEST:
        g = group_of(s)
        agg[g][0] += seen[s]
        agg[g][1] += pruned[s]
    return {g: 100 * v[1] / v[0] for g, v in agg.items()}, len(logs)


def damage(cfg):
    d, b = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(CSV_PATH)):
        if r["config"] != cfg:
            continue
        g = group_of(r["instance"])
        d[g].append(100 * float(r["delta"]))
        b[g].append(100 * float(r["mean_base"]))
    return {g: float(np.mean(v)) for g, v in d.items()}, {g: float(np.mean(v)) for g, v in b.items()}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), sharey="row")
    n_logs = {}
    for j, (cfg, label) in enumerate(CFGS):
        pr, n_logs[cfg] = pruning_rates(cfg)
        dm, base = damage(cfg)
        xs = np.arange(len(GROUPS))
        colors = [CLS_COLOR[c] for c, _ in GROUPS]
        ax = axes[0, j]
        vals = [pr[g] for g in GROUPS]
        ax.bar(xs, vals, color=colors, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=9, fontweight="bold")
        ax.set_title(label, fontsize=11.5)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_xticks(xs); ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel("Requests pruned (% of requests seen)", fontsize=10.5)
        ax = axes[1, j]
        vals = [dm[g] for g in GROUPS]
        ax.bar(xs, vals, color=colors, zorder=3)
        ax.axhline(0, color="#444444", linewidth=0.8)
        for x, v in zip(xs, vals):
            ax.annotate(("0" if round(v) == 0 else f"{v:+.0f}"), (x, v), textcoords="offset points", xytext=(0, 3 if v >= 0 else -11),
                        ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(-72, 24)  # 2026-10-12: more room below so the -60 label is not clipped
        ax.set_xticks(xs)
        # 2026-10-12: shorter tick labels (C1/C2 + geometry, baseline below) - the long form overlapped neighbours
        ax.set_xticklabels([f"C{c} {g}\nbase {base[(c, g)]:.0f}%" for c, g in GROUPS], fontsize=8.5)
        if j == 0:
            ax.set_ylabel("Service rate: Request-only minus Baseline (pp)", fontsize=10.5)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=CLS_COLOR["1"], label="class 1 (C1)"), Patch(fc=CLS_COLOR["2"], label="class 2 (C2)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=10.5)
    fig.suptitle("c2 mixedclass, Request Pruner only (t=0.5): pruning rate vs. damage by class and geometry",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
             "12 test instances (2 per group), Request-only vs. Baseline over 5 seeds; pruning rate pooled over seeds "
             f"(logs per config: {n_logs['bi200_ss100']} / {n_logs['bi400_ss200']} / {n_logs['bi400_ss80']}), "
             "all windows and epochs, incl. optimal_val pre-pass; epochs=5, lr=0.0001, pickup_slack off, min_retention 0.3",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c2_mixedclass_pruning_rate_vs_damage_by_class_geometry.{ext}", dpi=200)
    plt.close(fig)
    print(f"saved {OUT_DIR / 'c2_mixedclass_pruning_rate_vs_damage_by_class_geometry'}.png/.pdf")


if __name__ == "__main__":
    main()
