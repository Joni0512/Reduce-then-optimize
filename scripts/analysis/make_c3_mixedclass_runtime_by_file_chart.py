"""
2026-10-14: c3 MIXEDCLASS - training hours per training file, showing that five Class-2 files dominate the epoch time.
Source: scripts/analysis/profile_log_phases.py run on the cluster logs (2026-10-14), seed 42, epochs 5, lr 0.0001,
legacy scoring, no Request Pruner:
  Baseline batch_interval=200/step_size=100: sil_training_bi200_ss100_c3_mixedclass_legacy_seed42.log
      (cancelled job 5638523; epoch 1 plus the start of epoch 2, training total 42.03 h, 546 windows)
  Pair-only batch_interval=400/step_size=80: sil_training_bi400_ss80_c3_mixedclass_pairprunerv1final_pairt0.5_seed42.log
      (job 5638527, epoch 1 only, training total 144.73 h, 817 windows)
"other" = training total minus the five files shown (34 further files for the shared 39-file split minus these five).
Run: ./venv/bin/python3 scripts/analysis/make_c3_mixedclass_runtime_by_file_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["lc203", "lr203", "lr204", "lr206", "lr207"]
# training hours per file (profile_log_phases.py), total training hours
BASE_100 = ({"lc203": 18.93, "lr203": 15.16, "lr204": 0.50, "lr206": 1.80, "lr207": 0.20}, 42.03)
PAIR_80 = ({"lc203": 5.54, "lr203": 14.07, "lr204": 58.48, "lr206": 19.66, "lr207": 42.78}, 144.73)
NAVY, ORANGE = "#132A4C", "#F2A541"


def main():
    runs = [("Baseline, batch_interval=200/step_size=100", BASE_100, NAVY),
            ("Pair-only, batch_interval=400/step_size=80", PAIR_80, ORANGE)]
    cats = FILES + ["all other\ntraining files"]
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for k, (label, (d, total), color) in enumerate(runs):
        vals = [d[f] for f in FILES] + [total - sum(d.values())]
        heavy = sum(d.values())
        offs = (k - 0.5) * w
        bars = ax.bar(x + offs, vals, width=w, color=color, zorder=3,
                      label=f"{label}: {total:.0f} h training, {100 * heavy / total:.0f}% in these 5 files")
        for xi, v in zip(x + offs, vals):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5,
                        fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Training time (hours)", fontsize=10.5)
    ax.set_ylim(0, 72)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c3_mixedclass_runtime_by_file.{ext}", dpi=220)
    print(f"saved {OUT_DIR / 'c3_mixedclass_runtime_by_file'}.png/.pdf")


if __name__ == "__main__":
    main()
