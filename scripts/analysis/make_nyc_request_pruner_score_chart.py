import json
import numpy as np
import matplotlib.pyplot as plt

BLUE = "#164a87"
ORANGE = "#d67a2a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

with open(
    "/private/tmp/claude-501/-Users-joni-Desktop-Masterarbeit-Reduce-then-optimize/"
    "06426b27-f40c-4745-a015-657068376640/scratchpad/nyc_pruner_scores.json"
) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
fig.patch.set_facecolor("#fcfcfb")

bins = np.linspace(0.4, 1.0, 31)

for ax, split in zip(axes, ("val", "test")):
    ax.set_facecolor("#fcfcfb")
    scores = np.array(data[split]["scores"])
    labels = np.array(data[split]["labels"])

    # 2026-08-28: stacked (not alpha-overlaid) - the previous version drew
    # both histograms on top of each other with transparency, which reads as
    # muddy overlap. Stacking puts label=0 counts on the bottom and label=1
    # counts directly on top of them per bin, so every bar's total height is
    # still the full count at that score and the two groups are visually
    # separated with no color mixing.
    ax.hist(
        [scores[labels == 0], scores[labels == 1]],
        bins=bins, stacked=True, color=[ORANGE, BLUE],
        label=["label=0 (not assigned this window)", "label=1 (assigned this window)"],
        zorder=3, edgecolor="#fcfcfb", linewidth=0.3,
    )

    ax.axvline(0.5, color=TEXT_PRIMARY, linewidth=1.1, linestyle="--", zorder=4)
    ax.annotate("threshold=0.5", (0.5, 0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(4, 8), fontsize=9, color=TEXT_PRIMARY)

    n = len(scores)
    ax.set_title(f"{split.capitalize()} split (n={n})", fontsize=12.5, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("Predicted score (sigmoid output)", fontsize=10.5, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

axes[0].set_ylabel("Count", fontsize=10.5, color=TEXT_SECONDARY)

# 2026-08-28: figure-level legend instead of an in-axes one - an in-axes
# legend at "upper left" sat directly on top of the threshold annotation and
# the tallest bars, both cluttered and hard to read. Placed above the
# subplots, below the suptitle, where there's no data.
fig.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
    labels=["label=0 (not assigned this window)", "label=1 (assigned this window)"],
    loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=10, frameon=False,
)

fig.suptitle(
    "NYC RequestPruner MLP --- Score Distribution by True Label\n"
    "Winning sweep config (h=64, l=1, dropout=0.2, pos\\_weight=1.5, lr=0.01), 4min/8min step-size/horizon config",
    fontsize=12, color=TEXT_PRIMARY, y=1.06,
)

plt.tight_layout()
out_pdf = "figures_export/NYC/nyc_request_pruner_score_distribution.pdf"
out_png = "figures_export/NYC/nyc_request_pruner_score_distribution.png"
plt.savefig(out_pdf, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", out_pdf)
print("saved:", out_png)
