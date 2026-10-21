"""
2026-10-13: SRL training setup (old scheme: actor update after every window) as 4 boxes in a descending staircase layout (box 1 top left, boxes 2-4
each a bit further right and lower), for the SRL deck. Text = the bullets agreed in chat (English): setup, per epoch, per window, after the window loop;
pretraining target = immediate reward r_t (code mode "monte_carlo", no bootstrapping); actor-side Q = mean of the two target critics, TD target = min.
Writes figures_export/SRL/srl_setup_boxes.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
INK, GREY = "#2c2c2a", "#6b6b66"
boxes = [
    ("1  Setup", "#eeedfe", "#7F77DD", [
        "Actor randomly initialized (no SIL warm-start)",
        "Twin critic (two independent GAT critics)",
        "Critic pretraining: 10 epochs, actor frozen",
        "Pretraining target: immediate reward r_t (no bootstrapping)",
        "Train / val split: 38 / 9 instances"]),
    ("2  Per epoch (20 epochs)", "#e1f5ee", "#1D9E75", [
        "Training instances in random order",
        "Polyak update of both target critics before each instance",
        "Rolling horizon: batch_interval 200, step_size 100"]),
    ("3  Per window", "#fff3d6", "#D4A017", [
        "ILP decision using the actor scores",
        "Perturb-and-MAP: 20 perturbed candidates",
        "Q-values from the target critics (mean of the two)",
        "Softmax over candidates -> target action",
        "Fenchel-Young loss",
        "One Adam step on the actor"]),
    ("4  After the window loop", "#fbeaf0", "#CC79A7", [
        "End of instance: update of both online critics",
        "TD target: r_t + gamma * min(Q'_1, Q'_2)(next window)",
        "Reward: +1 per served request, known after the episode",
        "Validation every 5 epochs",
        "Best checkpoint selected by validation service rate"]),
]
W = 38.5
xs = [1.5, 42, 82.5, 123]
STEP = 8  # vertical drop per box (staircase)


def wrap(b, n=47):
    lines, cur = [], ""
    for w in b.split(" "):
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


heights = [8.2 + sum(2.5 * len(wrap(b)) + 1.4 for b in bl) + 0.6 for _, _, _, bl in boxes]
H = max(heights)
top0 = H + 6 + 3 * STEP
tops = [top0 - i * STEP for i in range(4)]
YMAX = top0 + 6
fig = plt.figure(figsize=(17, 17 * (YMAX / 163) * 0.92))
ax = fig.add_axes([0.005, 0.02, 0.99, 0.96]); ax.set_xlim(0, 163); ax.set_ylim(0, YMAX); ax.axis("off")
for i, ((title, fc, ec, bullets), x, top) in enumerate(zip(boxes, xs, tops)):
    ax.add_patch(FancyBboxPatch((x, top - H), W, H, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=1.8))
    ax.text(x + 1.8, top - 3.2, title, fontsize=12, weight="bold", color=INK, va="center")
    y = top - 8.2
    for b in bullets:
        lines = wrap(b)
        ax.text(x + 1.8, y, "\u2022", fontsize=10, color=ec, va="top", weight="bold")
        for k, line in enumerate(lines):
            ax.text(x + 3.6, y - k * 2.5, line, fontsize=9.3, color=INK, va="top")
        y -= 2.5 * len(lines) + 1.4
    if i < 3:
        ax.annotate("", xy=(xs[i + 1] - 0.3, tops[i + 1] - 4), xytext=(x + W + 0.3, top - 4),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, connectionstyle="arc3,rad=0.0"))
ax.text(1.5, YMAX - 2.5, "SRL training setup", fontsize=15, weight="bold", color=INK, va="center")
fig.text(0.01, 0.003, "Old scheme (actor update after every window). Twin critic: GAT, 4 heads, 2 layers; actor: MLP, hidden 64, 85 features; reward local_positive; gamma 0.99.", fontsize=7.5, color="#444444")
out = OUT / "srl_setup_boxes"
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved")
