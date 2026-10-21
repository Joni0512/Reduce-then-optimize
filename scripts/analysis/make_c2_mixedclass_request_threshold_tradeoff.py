"""
2026-10-13: c2 MIXEDCLASS, Request Pruner only: service rate vs. RUNTIME per threshold (trade-off view), one panel per
config. Each point = one threshold t = 0.1 ... 0.8 (mean of seeds 42 and 1); star = no-pruner baseline. Top-left =
fast and good. Combines the data of make_c2_mixedclass_request_threshold_bars.py (service rate) and
make_c2_mixedclass_request_threshold_runtime.py (runtime: sacct Elapsed / log timestamps, incl. post-training check).
Caveats: baseline runtime of bi200/ss100 and bi400/ss200 and t=0.5 runtime are seed 42 only; service rates are means of
the two seeds (see those scripts' docstrings).

Run: ./venv/bin/python3 scripts/analysis/make_c2_mixedclass_request_threshold_tradeoff.py
"""
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent))
from make_c2_mixedclass_request_threshold_bars import CONFIGS, DATA as SERVICE_DATA  # noqa: E402
from make_c2_mixedclass_request_threshold_runtime import BARS, COLORS as BAR_COLORS, DATA as RUNTIME_DATA, hours  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT_DIR = Path("figures_export/Request Pruner SIL/Cardinality 2")
COLORS = {CONFIGS[0]: "#276575", CONFIGS[1]: "#A6493A", CONFIGS[2]: "#3E8F6C"}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        pts = []
        for i, name in enumerate(BARS):
            sv = SERVICE_DATA[cfg][i]
            rt = RUNTIME_DATA[cfg][i]
            if sv is None or rt is None:
                continue
            pts.append((name, st.mean(hours(v) for v in rt), st.mean(sv[0])))
        base = pts[0]
        rest = pts[1:]
        # 2026-10-13: points coloured by threshold (palette of the bar charts) instead of a text label on every point -
        # the labels overlapped in the dense bi400 panels; only t=0.1 and t=0.8 are labelled, the rest via the legend.
        ax.plot([p[1] for p in rest], [p[2] for p in rest], "-", color="#888888", linewidth=1.3, zorder=2)
        for name, x, y in rest:
            ax.scatter([x], [y], s=70, color=BAR_COLORS[BARS.index(name)], edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.scatter([base[1]], [base[2]], marker="*", s=260, color="#1E2761", zorder=4)
        ax.axhline(base[2], color="#1E2761", linestyle="--", linewidth=1, alpha=0.5, zorder=1)
        for name, x, y in rest:
            # 2026-10-13: t=0.8 label moved to the left of its point (it sat on top of the dark point cluster)
            if name == "t=0.1":
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=9, color="#222222")
            elif name == "t=0.8":
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(-9, -4), ha="right", fontsize=9,
                            color="#222222")
        ax.annotate("baseline", (base[1], base[2]), textcoords="offset points", xytext=(8, 6), fontsize=8.5, color="#1E2761")
        ax.set_title(cfg, fontsize=11.5)
        ax.set_xlabel("Runtime (hours, incl. post-training check)", fontsize=10)
        ax.set_xlim(0, max(p[1] for p in pts) * 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(True, linestyle=":", alpha=0.4)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    handles = [Line2D([], [], marker="*", color="none", markerfacecolor="#1E2761", markersize=14, label="baseline")]
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=BAR_COLORS[i], markeredgecolor="#222222",
                       markersize=8, label=BARS[i]) for i in range(1, len(BARS))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=9, frameon=False, fontsize=9)
    axes[0].set_ylabel("Best val service rate (%)", fontsize=10.5)
    axes[0].set_ylim(20, 72)
    fig.suptitle("c2 mixedclass, Request Pruner only: service rate vs. runtime per threshold (t = 0.1 ... 0.8)", fontsize=12.5)
    fig.text(0.5, 0.005, "service rate = mean of seeds 42 and 1; runtime: sacct / log timestamps (baseline bi200/ss100, bi400/ss200 and t=0.5: "
             "seed 42 only); epochs=5, lr=0.0001, pickup_slack off, min_retention 0.3; star = no pruner",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"c2_mixedclass_request_threshold_tradeoff.{ext}", dpi=200)
    print("saved c2_mixedclass_request_threshold_tradeoff.png/.pdf")


if __name__ == "__main__":
    main()
