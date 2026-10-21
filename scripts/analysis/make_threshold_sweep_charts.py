"""
2026-09-26: Threshold sweep bar charts (0.3/0.4/0.5 vs. no-pruner baseline) for bi200/ss100,
both class1 and mixed-class - Service Rate and Runtime, 2 panels each (class1 | mixed-class).
All prefix-label Request Pruner only (no Pair Pruner), min_retention_fraction=0.3 throughout.

Sources (2026-09-25/26, "Best val service rate" + start/"Run complete" timestamps, 5 seeds
each: 42,1,2,3,4):
- class1: baseline 77.55(std1.15)/95.6(no std, single extrapolated mean); t=0.5 65.85(2.45)/49.4(1.4);
  t=0.4 73.65(2.20)/69.5(3.9); t=0.3 77.55(0.88)/87.5(4.9)
- mixed-class: baseline 64.85(3.04)/279.4(2.6); t=0.5 33.35(1.68)/109.2(3.5);
  t=0.4 36.76(3.57)/124.7(7.2); t=0.3 44.56(8.29)/149.5(6.8)

Run: python3 -m scripts.analysis.make_threshold_sweep_charts
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "figures_export/Request Pruner SIL"
VARIANTS = ["baseline", "t0.5", "t0.4", "t0.3"]
LABELS = ["No pruner\n(baseline)", "threshold\n=0.5", "threshold\n=0.4", "threshold\n=0.3"]
COLORS = ["#1E2761", "#A6493A", "#D68C3B", "#3E8F6C"]

SERVICE = {
    "class1": {"baseline": (77.55, 1.15), "t0.5": (65.85, 2.45), "t0.4": (73.65, 2.20), "t0.3": (77.55, 0.88)},
    "mixed-class": {"baseline": (64.85, 3.04), "t0.5": (33.35, 1.68), "t0.4": (36.76, 3.57), "t0.3": (44.56, 8.29)},
}
RUNTIME = {
    "class1": {"baseline": (95.6, None), "t0.5": (49.4, 1.4), "t0.4": (69.5, 3.9), "t0.3": (87.5, 4.9)},
    "mixed-class": {"baseline": (279.4, 2.6), "t0.5": (109.2, 3.5), "t0.4": (124.7, 7.2), "t0.3": (149.5, 6.8)},
}


def make_chart(data, ylabel, title, out_stem, ylim, pct=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, cls in zip(axes, ["class1", "mixed-class"]):
        vals = [data[cls][v][0] for v in VARIANTS]
        errs = [data[cls][v][1] if data[cls][v][1] is not None else 0 for v in VARIANTS]
        x = np.arange(len(VARIANTS))
        ax.bar(x, vals, yerr=errs, capsize=5, color=COLORS, width=0.65, zorder=3)
        for xi, v, s in zip(x, vals, errs):
            label = f"{v:.1f}" + (f" ± {s:.1f}" if s else "")
            ax.annotate(label, (xi, v + s), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.set_title(cls, fontsize=13)
        ax.set_ylim(*ylim)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=11)
    fig.suptitle(f"{title}\nbi200/ss100, Request Pruner only, min_retention_fraction=0.3, n=5 seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{out_stem}.png", dpi=200)
    fig.savefig(f"{OUT_DIR}/{out_stem}.pdf")
    print(f"saved {OUT_DIR}/{out_stem}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_chart(SERVICE, "Best val service rate (%)", "Threshold Sweep: Service Rate",
               "threshold_sweep_service_rate", ylim=(0, 100), pct=True)
    make_chart(RUNTIME, "Runtime (minutes, full 5-epoch training run)", "Threshold Sweep: Runtime",
               "threshold_sweep_runtime", ylim=(0, 320))
