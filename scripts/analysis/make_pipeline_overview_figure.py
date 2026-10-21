"""
2026-10-05: Overview figure of the Reduce-then-Optimize pipeline (thesis chapter 04_Method).

Flow verified against code (rtv_solver/handlers/trip_handler.py TripHandler.run(), lines 66-176):
  active requests of the rolling-horizon window
  -> RequestPruner (MLP on per-request features; force-keeps committed + urgent requests,
     min-retention floor; pruned requests are NOT rejected, they stay in state for the next window)
  -> RR graph over the REMAINING requests -> RequestGraphPruner (GNN) -> allowed request pairs
  -> RTV generation (single trips -> shared trips up to max_cardinality, only allowed pairs)
  -> trip scoring model (learned, SIL) -> score-maximizing ILP (CO layer) -> assignment.
Both pruners are optional and independent (USE_REQUEST_PRUNER / USE_REQUEST_GRAPH_PRUNER).

Run: ./venv/bin/python3 scripts/analysis/make_pipeline_overview_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT_DIR = Path("figures_export/04_Method")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEARNED = "#276575"
COMBI = "#1E2761"
NEUTRAL = "#E9ECEF"
NEUTRAL_EDGE = "#8A93A0"

# (title, subtitle, kind, optional)
BOXES = [
    ("Active requests", "rolling-horizon\nwindow", "neutral", False),
    ("Request Pruner", "MLP on per-request\nfeatures", "learned", True),
    ("RR graph", "remaining requests,\nall pairs", "neutral", False),
    ("Pair Pruner", "GNN on RR graph", "learned", True),
    ("RTV generation", "single → shared trips\n(allowed pairs only)", "combi", False),
    ("Trip scoring", "learned model\n(imitation learning)", "learned", False),
    ("ILP", "score maximization\n(CO layer)", "combi", False),
    ("Assignment", "committed trips", "neutral", False),
]

W, H, GAP = 1.62, 1.45, 0.42
# 2026-10-12: figure made flatter for the slide (was figsize 16 x 4.9, Y0 1.9): Y0 lowered, texts under the boxes
# shortened to one line each, red note shortened, legend in one row. Content/order of the pipeline unchanged.
Y0 = 0.95


def draw_box(ax, x, title, sub, kind, optional):
    if kind == "neutral":
        fc, ec, tc = NEUTRAL, NEUTRAL_EDGE, "#222222"
    elif kind == "learned":
        fc, ec, tc = LEARNED, LEARNED, "white"
    else:
        fc, ec, tc = COMBI, COMBI, "white"
    patch = FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0.02,rounding_size=0.14",
                           fc=fc, ec=ec, lw=1.8, zorder=3,
                           ls=(0, (4, 2)) if optional else "-")
    ax.add_patch(patch)
    ax.text(x + W / 2, Y0 + H * 0.66, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=tc, zorder=4)
    ax.text(x + W / 2, Y0 + H * 0.30, sub, ha="center", va="center", fontsize=8, color=tc, zorder=4)


def arrow(ax, x1, x2, y, **kw):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color="#444444", zorder=2, **kw))


def main():
    n = len(BOXES)
    total_w = n * W + (n - 1) * GAP
    fig, ax = plt.subplots(figsize=(16, 3.4))
    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(0.0, 3.45)
    ax.axis("off")

    xs = [i * (W + GAP) for i in range(n)]
    for x, (t, s, k, o) in zip(xs, BOXES):
        draw_box(ax, x, t, s, k, o)
    for i in range(n - 1):
        arrow(ax, xs[i] + W + 0.03, xs[i + 1] - 0.03, Y0 + H / 2)

    # effect labels under the two pruners
    ax.text(xs[1] + W / 2 + 0.25, Y0 - 0.12, "keeps urgent + committed requests", ha="left", va="top",
            fontsize=7.8, color="#444444")
    ax.text(xs[3] + W / 2, Y0 - 0.12, "allowed pairs  E′ ⊆ E", ha="center", va="top",
            fontsize=7.8, color="#444444")

    # pruned requests are not rejected: dashed feedback arrow from the Request Pruner
    # (left bottom corner) back to the request pool (2026-10-05: now attached to both boxes)
    ax.add_patch(FancyArrowPatch((xs[1] + 0.4, Y0 - 0.02), (xs[0] + W / 2, Y0 - 0.02),
                                 connectionstyle="arc3,rad=-0.4", arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, ls=(0, (4, 3)), color="#A6493A", zorder=1))
    ax.text((xs[0] + W / 2 + xs[1] + 0.4) / 2, Y0 - 0.55,
            "pruned requests are deferred, not rejected", ha="center", va="center", fontsize=8,
            color="#A6493A")

    # grouping brackets above the row
    ax.annotate("", xy=(xs[1], Y0 + H + 0.35), xytext=(xs[3] + W, Y0 + H + 0.35),
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#666666"))
    ax.text((xs[1] + xs[3] + W) / 2, Y0 + H + 0.5, "Reduce", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#444444")
    ax.annotate("", xy=(xs[4], Y0 + H + 0.35), xytext=(xs[6] + W, Y0 + H + 0.35),
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#666666"))
    ax.text((xs[4] + xs[6] + W) / 2, Y0 + H + 0.5, "Optimize", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#444444")

    ax.legend(handles=[
        Patch(fc=LEARNED, ec=LEARNED, label="learned component"),
        Patch(fc=COMBI, ec=COMBI, label="combinatorial / exact component"),
        Patch(fc="white", ec="#444444", ls="--", label="optional module"),
    ], loc="lower right", bbox_to_anchor=(1.0, -0.03), fontsize=8.5, frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "reduce_then_optimize_pipeline.png", dpi=220)
    fig.savefig(OUT_DIR / "reduce_then_optimize_pipeline.pdf")
    print(f"saved {OUT_DIR / 'reduce_then_optimize_pipeline'}.png/.pdf")


if __name__ == "__main__":
    main()
