"""
2026-10-10: schematic of the PROPOSED joint-buffer scheme ("buffered_joint", not implemented yet, see chat): one shared FIFO buffer over 10 epochs for actor and
critic; per epoch (A) collect with the frozen actor, (B) U update steps (~23, like today's actor steps): sample a batch with replacement ->
actor update -> several critic updates -> Polyak. Parameter values (10 epochs ~3600 windows, batch 16, 6-7 critic steps per actor step,
critic batch 12, tau ~0.0083 per step) are the proposed defaults, still open for confirmation. PNG + PDF in figures_export/SRL/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
PURPLE, TEAL, GRAY, BOX, TXT, BUF = "#7F77DD", "#1D9E75", "#9a9a94", "#f1efe8", "#2c2c2a", "#fff3d6"

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 130); ax.set_ylim(0, 90); ax.axis("off")


def box(x, y, w, h, fc=BOX, ec="#b4b2a9", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def t(x, y, s, fs=9, weight="normal", color=TXT, ha="left", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, weight=weight, color=color, ha=ha, va=va, style=style)


def arrow(x0, y0, x1, y1, color="#6b6b66", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


# title
t(1, 87, "Proposed joint buffer: actor and critic update from one shared buffer (per epoch)", fs=13, weight="bold")

# ---- Phase A
t(1, 80.5, "Phase A - collect (actor frozen at W_e)", fs=10.5, weight="bold")
for k, x in enumerate((1, 22, 43)):
    box(x, 66, 19, 11)
    t(x + 1.2, 74.3, f"Instance {k + 1}", fs=8.5)
    for i in range(4):
        ax.add_patch(Circle((x + 3 + i * 3.4, 69.5), 1.15, fc=GRAY, ec="none"))
    ax.add_patch(Rectangle((x + 15.2, 68), 3, 3, fc=TEAL, ec="none"))
t(65, 71.5, "... 38 instances", fs=8.5)
t(1, 63, "per window: ILP decision from the frozen actor -> store entry = snapshot (actor) + critic graph; reward and next window added at instance end", fs=8.3, style="italic")
t(1, 60.7, "critics: as today only the end-of-instance rewards are computed here (the critic learns in Phase B)", fs=8.3, style="italic")
arrow(85, 71.5, 93, 71.5)

# ---- shared buffer
box(94, 56, 35, 25, fc=BUF, ec="#d4a017", lw=1.6)
t(95.5, 78, "Shared buffer (FIFO)", fs=10, weight="bold")
t(95.5, 75.2, "capacity: 10 epochs (~3600 windows)", fs=8.3)
for i in range(6):
    ax.add_patch(Rectangle((96 + i * 5.4, 62.5), 4.7, 9, fc="#fde9a8", ec="#d4a017", lw=0.8))
    t(98.35 + i * 5.4, 67, f"e-{5 - i}" if i < 5 else "e", fs=7.5, ha="center")
t(95.5, 59.2, "epoch 1: 1 epoch of windows, epoch 2: 2 epochs, ...", fs=7.6)
t(95.5, 57.3, "oldest epoch is dropped when full", fs=7.6)
arrow(107, 55.5, 107, 49.5)

# ---- Phase B
t(1, 52.5, "Phase B - update, repeated U ~ 23 times per epoch (same number of actor steps as today)", fs=10.5, weight="bold")
steps = [
    (1, "1  Sample", "16 windows from the buffer,\nwith replacement"),
    (25.5, "2  Actor update", "theta = actor(features), 20 perturbed\ncandidates -> target-critic Q -> softmax\n-> target action -> Fenchel-Young loss,\nmean over the batch, 1 Adam step (clip 1.0)"),
    (62, "3  Critic update (6-7 times)", "sample 12 windows (with replacement),\nTD target r + gamma * min(Q_target1, Q_target2),\nMSE, Adam step for critic 1 and critic 2"),
    (99, "4  Polyak", "target critics <- tau * critic\n+ (1 - tau) * target,\ntau ~ 0.0083 per step"),
]
for x, head, body in steps:
    w = 22 if x < 25 else (35 if x < 60 else (35 if x < 95 else 30))
    w = {1: 22, 25.5: 34, 62: 35, 99: 30}[x]
    box(x, 24, w, 24)
    t(x + 1.4, 44.8, head, fs=9.5, weight="bold")
    t(x + 1.4, 35, body, fs=7.9, va="center")
arrow(23.2, 36, 25.3, 36); arrow(59.7, 36, 61.8, 36); arrow(97.2, 36, 98.8, 36)
ax.add_patch(Circle((12, 28.5), 1.3, fc=PURPLE, ec="none")); t(14.5, 28.5, "actor step", fs=8)
ax.add_patch(Rectangle((40, 27), 3, 3, fc=TEAL, ec="none")); t(44.5, 28.5, "(critics only evaluate)", fs=7.8)
ax.add_patch(Rectangle((70, 27), 3, 3, fc=TEAL, ec="none")); t(74.5, 28.5, "critic steps", fs=7.8)

# loop arrow
ax.annotate("", xy=(2, 22.8), xytext=(113, 22.8), arrowprops=dict(arrowstyle="-|>", color="#6b6b66", lw=1.2, connectionstyle="arc3,rad=0"))
t(46, 21, "repeat U times", fs=8, style="italic")

# result
arrow(65, 20, 65, 15.5)
box(26, 7, 78, 8)
t(65, 11, "new actor W_(e+1) and new critics -> next epoch collects with the updated actor; the buffer keeps the old epochs", fs=9, ha="center")

# legend
ax.add_patch(Circle((3, 2.7), 1.15, fc=GRAY, ec="none")); t(5.5, 2.7, "window stored (actor frozen)", fs=8)
ax.add_patch(Circle((36, 2.7), 1.3, fc=PURPLE, ec="none")); t(38.5, 2.7, "actor step", fs=8)
ax.add_patch(Rectangle((55, 1.2), 3, 3, fc=TEAL, ec="none")); t(59.5, 2.7, "critic update", fs=8)
fig.text(0.01, 0.005, "Proposal (not implemented): parameters are proposed defaults (10 epochs ~3600 windows, batch 16, 6-7 critic steps per actor step, critic batch 12, tau ~0.0083). "
         "Today's scheme: buffer = 1 epoch, no replacement, actor-only update phase, critic trained per instance.", fontsize=6.5, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = OUT / "srl_joint_buffer_scheme"
fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved")
