"""
2026-10-14: slide graphic "Idea: reward shaping for urgent requests" (3 boxes side by side). Numbers computed from the rows of
scripts/analysis/make_all_finished_runs_table.py (figures_export/SRL/all_finished_runs_table.*), state 2026-10-14: 34 buffered 20-epoch runs without collapse,
41 buffered runs, 14 old per-window runs, 47 rows with a test value. Counts are descriptive (single runs, different hyperparameters). Writes
figures_export/SRL/srl_reward_shaping_slide.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
INK, GREY = "#2c2c2a", "#6b6b66"
boxes = [
    ("Current reward (local_positive)", "#eeedfe", "#7F77DD", [
        "+1 per served request, booked in the window where its pickup deadline passes",
        "Computed after the episode (needs the final service outcome)",
        "Urgent requests give their +1 immediately, non-urgent ones later",
        "Scale is 10 to 50 times larger than local (-1 per missed request); gamma, Polyak rate and critic_lr were never re-tuned for it"]),
    ("Why look at it", "#e1f5ee", "#1D9E75", [
        "RHO with urgency penalty (rejection penalty x2 or x4 for requests that expire before the next decision): val 0.809",
        "RHO without: 0.704, SRL (config D, 5 seeds): 0.718",
        "SRL wins on lr202, loses on lc202",
        "Hypothesis: the gap is not only a question of the reward form (RHO enforces urgency by a rule in the ILP)"]),
    ("Ideas (nothing implemented yet)", "#fff3d6", "#D4A017", [
        "Weight +w for requests with tight time windows (static property)",
        "Penalty -c for lost requests (combine local and local_positive)",
        "Potential-based shaping: only moves credit in time, does not change which action is better",
        "Not: a bonus for last-minute service (rewards waiting)",
        "Urgency can also enter as a feature or an ILP bonus, outside the reward"]),
]


def wrap(text, n):
    lines, cur = [], ""
    for w in text.split(" "):
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


W, GAP, X0, TOP = 49, 3.5, 2, 72
heights = [8.5 + sum(2.7 * len(wrap(b, 47)) + 1.7 for b in bl) + 0.5 for _, _, _, bl in boxes]
H = max(heights)
fig = plt.figure(figsize=(16, 7.4))
ax = fig.add_axes([0.005, 0.03, 0.99, 0.95]); ax.set_xlim(0, 160); ax.set_ylim(TOP - H - 6, TOP + 8); ax.axis("off")
ax.text(X0, TOP + 5, "Idea: reward shaping for urgent requests", fontsize=16, weight="bold", color=INK, va="center")
for i, (title, fc, ec, bullets) in enumerate(boxes):
    x = X0 + i * (W + GAP)
    ax.add_patch(FancyBboxPatch((x, TOP - H), W, H, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=1.8))
    ax.text(x + 1.8, TOP - 3.5, title, fontsize=12.5, weight="bold", color=INK, va="center")
    y = TOP - 8.5
    for b in bullets:
        lines = wrap(b, 47)
        ax.text(x + 1.8, y, "•", fontsize=11, color=ec, va="top", weight="bold")
        for k, line in enumerate(lines):
            ax.text(x + 3.8, y - k * 2.7, line, fontsize=10, color=INK, va="top")
        y -= 2.7 * len(lines) + 1.7
ax.text(X0, TOP - H - 3.2, "Sources: mc_return_builder.py (build_local_positive), RHO urgency runs outputs/rho_val_runs/*_urgent{2,4} (3 runs each), SRL config D bestD_s301-305. Any shaping needs a rescaled reward (e.g. divided by the number of requests) and re-tuned critic_lr / gamma.", fontsize=7.8, color="#444444", va="center")
out = OUT / "srl_reward_shaping_slide"
fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved")
