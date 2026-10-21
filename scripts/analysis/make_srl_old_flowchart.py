"""
2026-10-13: flowchart of the OLD SRL training scheme (actor update after every rolling-horizon window, no actor buffer) for the SRL deck.
Content follows the structure the user pasted (random actor init, critic pretrain 10 epochs, 20 epochs with instance/window loops, Polyak before each
instance, per-window actor update via perturb-and-MAP -> target-critic Q -> softmax -> target action -> Fenchel-Young -> one Adam step, critic update at
instance end from the replay buffer, validation every 5 epochs). Numbers (358 steps/epoch, ceil(windows/3) ~ 4 updates, replay 40/batch 12) as in the
chat, approximate. Writes figures_export/SRL/srl_old_scheme_flowchart.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
INK, GREY, PURPLE, TEAL, AMBER = "#2c2c2a", "#6b6b66", "#7F77DD", "#1D9E75", "#D4A017"
fig = plt.figure(figsize=(16, 8.2))
ax = fig.add_axes([0.005, 0.04, 0.99, 0.95])
ax.set_xlim(0, 160); ax.set_ylim(0, 84); ax.axis("off")


def box(x, y, w, h, fc="#f1efe8", ec="#b4b2a9", lw=1.2, ls="-", r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def txt(x, y, s, fs=9, w="normal", c=INK, ha="center", va="center", st="normal"):
    ax.text(x, y, s, fontsize=fs, weight=w, color=c, ha=ha, va=va, style=st, linespacing=1.35)


def arr(x0, y0, x1, y1, c=GREY, lw=1.6, rad=0.0, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=c, lw=lw, connectionstyle=f"arc3,rad={rad}"))


txt(2, 81, "SRL training scheme (old version: actor update after every window)", fs=14, w="bold", ha="left")

# top row
box(2, 66, 30, 10); txt(17, 71, "Random actor init\n(no SIL warm-start)", fs=9.5, w="bold")
box(40, 66, 44, 10); txt(62, 71, "Phase 0: critic pretrain\n10 epochs, actor frozen", fs=9.5, w="bold")
arr(32, 71, 40, 71)
arr(84, 71, 92, 71)
txt(113, 71, "Phase 1: 20 epochs", fs=10, w="bold", ha="center")
arr(113, 68, 113, 62.5)

# epoch loop container
box(2, 3, 156, 59, fc="#fbfaf6", ec=GREY, lw=1.4, ls="--", r=2)
txt(5, 59, "for epoch = 1 .. 20", fs=10.5, w="bold", ha="left")

# instance loop container
box(5, 8, 108, 47, fc="#f6f4ec", ec=GREY, lw=1.2, ls="--", r=2)
txt(8, 52, "for each training instance (38, random order)", fs=9.5, w="bold", ha="left")

# polyak
box(8, 25, 19, 20); txt(17.5, 38, "Polyak update", fs=9.5, w="bold"); txt(17.5, 31, "target critics\ntau = 0.005", fs=8.5)
arr(27, 35, 31, 35)

# window loop container
box(31, 12, 52, 36, fc="#ffffff", ec=PURPLE, lw=1.4, ls="--", r=2)
txt(33.5, 45, "for each window (step_size = 100)", fs=9, w="bold", c=PURPLE, ha="left")
box(34, 20, 16, 20, fc="#eeedfe", ec=PURPLE); txt(42, 36, "ILP decision", fs=9, w="bold"); txt(42, 28.5, "actor scores\nthe candidate\ntrips", fs=8.5)
arr(50, 30, 53, 30, c=PURPLE)
box(53, 15, 28, 27, fc="#eeedfe", ec=PURPLE)
txt(67, 38.5, "Actor update", fs=9.5, w="bold")
txt(67, 27, "perturb theta (20 samples)\n-> target-critic Q\n-> softmax (tau = 0.02)\n-> target action\n-> Fenchel-Young loss\n-> 1 Adam step (clip 1.0)", fs=8.2)
arr(67, 15, 42, 15, c=PURPLE, rad=-0.0)  # loop back along the bottom
arr(42, 15, 42, 20, c=PURPLE)
txt(55, 13.2, "next window (~358 actor steps per epoch)", fs=7.8, c=PURPLE, st="italic")

# critic update
arr(83, 30, 87, 30)
box(87, 17, 24, 27, fc="#e1f5ee", ec=TEAL)
txt(99, 39.5, "Critic update", fs=9.5, w="bold"); txt(99, 37, "(end of instance)", fs=8, c=GREY)
txt(99, 28, "replay buffer (40)\nceil(windows/3) ~ 4 updates\nbatch 12, TD target\n(min of 2 target critics)", fs=8.2)
# loop back instance
arr(99, 17, 99, 10.5, c=GREY); arr(99, 10.5, 17.5, 10.5, c=GREY); arr(17.5, 10.5, 17.5, 25, c=GREY)
txt(58, 9.2, "next instance", fs=8, c=GREY, st="italic")

# validation
arr(113, 35, 119, 35)
box(119, 17, 36, 27, fc="#fff3d6", ec=AMBER, lw=1.5)
txt(137, 39.5, "Validation", fs=9.5, w="bold")
txt(137, 28, "every 5 epochs\n9 validation instances,\nno exploration\nsave best checkpoint (by val)", fs=8.5)
arr(137, 17, 137, 6.5, c=GREY, style="-"); arr(137, 6.5, 3.5, 6.5, c=GREY, style="-"); arr(3.5, 6.5, 3.5, 40, c=GREY, style="-"); arr(3.5, 40, 5.2, 40, c=GREY)
txt(70, 5.0, "next epoch", fs=8, c=GREY, st="italic")
txt(136, 49, "after epoch 20:\nreload best checkpoint", fs=8.5, c=GREY, st="italic")

fig.text(0.01, 0.005, "Twin critic (GAT, 4 heads, 2 layers), actor MLP (hidden 64, 85 features), gamma 0.99, reward local_positive, bi200/ss100. Counts (358 steps/epoch, ~4 critic updates/instance) are approximate.",
         fontsize=7, color="#444444")
out = OUT / "srl_old_scheme_flowchart"
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved")
