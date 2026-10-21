"""
2026-09-26: c3 counterpart to make_pruner_score_distribution.py - shows WHY the c3
Request Pruner (expert-prefix-label, k=3, t=0.5) prunes almost nothing live (0-1%,
see chat 2026-09-26): raw sigmoid scores (val+test) are compressed almost entirely
above the threshold, and the underlying label positivity rate is already >50%.

Uses score .npz dumps produced by check_c3_pruner_near_noop.py (run on the cluster,
where the c3 datasets/checkpoints live, then rsynced down - see chat 2026-09-26).

Run: ./venv/bin/python3 scripts/analysis/make_c3_pruner_noop_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_DIR = Path("outputs/c3_pruner_noop_check")
OUT_DIR = "figures_export/Request Pruner SIL"

CONFIGS = ["bi200_ss100", "bi400_ss200", "bi400_ss80"]
LABELS = ["bi200/ss100", "bi400/ss200", "bi400/ss80"]
COLORS = ["#276575", "#A6493A", "#3E8F6C"]
THRESHOLD = 0.5

# live pruning rate measured directly from SIL logs (mean of seed42/seed1, see chat 2026-09-26)
LIVE_PRUNE_RATE = {"bi200_ss100": 0.0, "bi400_ss200": 1.045, "bi400_ss80": 0.345}


def make_chart():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, cfg, label, color in zip(axes, CONFIGS, LABELS, COLORS):
        z = np.load(IN_DIR / f"{cfg}_scores.npz")
        scores = z["scores"]
        n_pos, n_total = int(z["label_pos"]), int(z["label_total"])
        pos_rate = 100 * n_pos / n_total

        mean_s, median_s, min_s = scores.mean(), np.median(scores), scores.min()
        below_t = np.mean(scores < THRESHOLD) * 100

        ax.hist(scores, bins=40, range=(0, 1), color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"threshold={THRESHOLD}")
        ax.axvline(mean_s, color="#E8A33D", linestyle="-", linewidth=2, label=f"mean={mean_s:.3f}")
        ax.axvline(min_s, color="#5A5A5A", linestyle=":", linewidth=2, label=f"min={min_s:.3f}")

        ax.set_title(f"{label}\nlabel positive={pos_rate:.1f}%, score<0.5={below_t:.2f}%\n"
                     f"live pruning rate={LIVE_PRUNE_RATE[cfg]:.2f}%", fontsize=10.5)
        ax.set_xlabel("Sigmoid score", fontsize=10.5)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Count (val+test rows)", fontsize=10.5)
    fig.suptitle("c3 Request Pruner (expert-prefix-label, $k{=}3$): scores are compressed above threshold\n"
                 "-> the pruner is a near no-op, not a calibration success", fontsize=12.5)
    fig.tight_layout()

    fig.savefig(f"{OUT_DIR}/c3_pruner_score_distribution_noop.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/c3_pruner_score_distribution_noop.pdf")
    print(f"saved {OUT_DIR}/c3_pruner_score_distribution_noop.{{png,pdf}}")


if __name__ == "__main__":
    make_chart()
