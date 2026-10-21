"""
2026-10-14: slide graphic "Observations from all finished runs" (3 boxes side by side). Numbers computed from the rows of
scripts/analysis/make_all_finished_runs_table.py (figures_export/SRL/all_finished_runs_table.*), state 2026-10-14: 34 buffered 20-epoch runs without collapse,
41 buffered runs, 14 old per-window runs, 47 rows with a test value. Counts are descriptive (single runs, different hyperparameters). Writes
figures_export/SRL/srl_observations_slide.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures_export" / "SRL"
INK, GREY = "#2c2c2a", "#6b6b66"
boxes = [
    ("What drives the results", "#eeedfe", "#7F77DD", [
        "Config matters most: D ends at 0.70 to 0.74 (5 seeds), E at 0.54 to 0.74",
        "Seed spread of E is about 5x larger than of D",
        "Top sweep values (0.739 to 0.740) did not repeat: E over 5 seeds has best val 0.650 ± 0.079"]),
    ("Training dynamics", "#e1f5ee", "#1D9E75", [
        "Peak often not at the end: best val is 0.027 above val at epoch 20 on average",
        "Best epoch: 20 in 15 runs, 15 in 12, 10 in 3, 5 in 4 (34 runs)",
        "Dips mid-training: 3 of 34 runs drop by more than 0.05 at epoch 15 and recover",
        "More epochs do not help reliably: E peaks at epoch 25, D (seed 42) drops from 0.761 to 0.616 at epoch 30"]),
    ("Stability and evaluation", "#fff3d6", "#D4A017", [
        "Collapses (val 0.0): old method 4 of 14 runs, new method 1 of 41",
        "Test is above val in 77 % of runs (mean +0.021); RHO shows the same (0.704 val, 0.738 test)",
        "Compare val with val and test with test only"]),
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
ax.text(X0, TOP + 5, "Observations from all finished runs", fontsize=16, weight="bold", color=INK, va="center")
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
ax.text(X0, TOP - H - 3.2, "Counts from the table of all finished runs (34 buffered 20-epoch runs without collapse, 41 buffered and 14 old runs, 47 rows with a test value); single runs with different "
        "hyperparameters, descriptive only.", fontsize=7.8, color="#444444", va="center")
out = OUT / "srl_observations_slide"
fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved")
