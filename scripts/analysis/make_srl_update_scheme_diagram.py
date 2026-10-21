"""
2026-10-09: schematic of one training epoch - OLD per-window actor update vs. NEW buffered "collect, then update" scheme (see chat).
Purely illustrative (3 of 38 instances drawn, dot counts are symbolic: a real instance has ~9-10 windows, an epoch ~360 windows ->
~23 actor steps of batch 16). Facts: old = one actor step after every rolling-horizon window (step_size) and one critic update block at
the end of each instance; new = phase A actor frozen (windows stored, critic learns as before), phase B windows shuffled, batches of 16,
mean loss, one clipped Adam step per batch, critics frozen. Writes PNG + PDF into figures_export/SRL/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
PURPLE, TEAL, GRAY, BOX, TXT = "#7F77DD", "#1D9E75", "#9a9a94", "#f1efe8", "#2c2c2a"

fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_xlim(0, 120); ax.set_ylim(0, 72); ax.axis("off")


def box(x, y, w, h, label=None, fc=BOX):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="#b4b2a9", lw=1))
    if label:
        ax.text(x + 1.5, y + h - 3.2, label, fontsize=9, color=TXT, va="center")


def dots(x0, y, n, color, step, r=1.5):
    for i in range(n):
        ax.add_patch(Circle((x0 + i * step, y), r, fc=color, ec="none"))


def square(x, y, s=3.4):
    ax.add_patch(Rectangle((x, y - s / 2), s, s, fc=TEAL, ec="none"))


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#6b6b66", lw=1.3))


# ---- old method
ax.text(1, 69, "Old method: actor and critic alternate inside the epoch", fontsize=12, weight="bold", color=TXT)
for k, x in enumerate((2, 36, 70)):
    box(x, 50, 31, 13, f"Instance {k + 1}")
    dots(x + 4, 54, 5, PURPLE, 4.6)
    square(x + 25.5, 54)
ax.text(104, 58, "... 38\ninstances", fontsize=9, color=TXT, va="center")
arrow(14, 47.5, 14, 52); ax.text(2, 45.5, "actor step after every window", fontsize=9, color=TXT)
arrow(63.2, 47.5, 63.2, 52); ax.text(46, 45.5, "critic update at instance end", fontsize=9, color=TXT)

# ---- new method
ax.text(1, 36, "New method: collect, then update", fontsize=12, weight="bold", color=TXT)
ax.text(2, 31.5, "Phase A: actor frozen, windows stored, critic learns as before", fontsize=9.5, color=TXT)
ax.text(66, 31.5, "Phase B: critics frozen, actor learns", fontsize=9.5, color=TXT)
for k, x in enumerate((2, 24, 46)):
    box(x, 17, 20, 12, f"Instance {k + 1}")
    dots(x + 3, 21, 4, GRAY, 3.4, 1.2)
    square(x + 15.6, 21, 3.0)
ax.text(70, 22.5, "... 38 instances", fontsize=8.5, color=TXT, va="center")
arrow(88.5, 23, 93, 23)
box(95, 15, 24, 15, None)
ax.text(96.5, 27, "shuffle, batches of 16", fontsize=8.5, color=TXT, va="center")
dots(98, 21, 4, PURPLE, 4.6, 1.5)
ax.text(96.5, 17.2, "mean loss, 1 Adam step per\nbatch (~23 steps / epoch)", fontsize=7.2, color=TXT, va="center")
ax.text(2, 11, "Next epoch starts with the updated actor.", fontsize=9, color=TXT, style="italic")

# ---- legend
ax.add_patch(Circle((3, 4), 1.5, fc=PURPLE, ec="none")); ax.text(6, 4, "actor step", fontsize=9, va="center", color=TXT)
square(24, 4); ax.text(29, 4, "critic update (TD, replay batch 12)", fontsize=9, va="center", color=TXT)
ax.add_patch(Circle((62, 4), 1.2, fc=GRAY, ec="none")); ax.text(65, 4, "window stored (actor frozen)", fontsize=9, va="center", color=TXT)
fig.text(0.01, 0.005, "Schematic, 3 of 38 instances drawn; dot counts symbolic (real: ~9-10 windows per instance, ~360 windows per epoch). batch_interval=200/step_size=100. "
         "Source: coaml_pipeline.py, srl_training_loop.py.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_update_scheme_old_vs_new"
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved", out)
