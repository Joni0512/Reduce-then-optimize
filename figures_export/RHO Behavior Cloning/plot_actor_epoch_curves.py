"""Actor (step_size=100/batch_interval=200, from scratch) state-sync BC: val / overfit_check / FY loss over epochs,
teacher RHO batch_interval=400 vs 600 (step_size=100). Values copied from
state_sync_test_4_scratch.log (teacher 400) and state_sync_test_5_rho600_scratch.log (teacher 600), 2026-09-27/28.
RHO reference lines = mean of 3 RHO runs on the same 9 VAL_INSTANCES (see plot_rho_horizon.py)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__file__).resolve().parent
ep_val = [5, 10, 15, 20]
runs = {
  "teacher RHO batch_interval=400/step_size=100": dict(
      val=[0.6655, 0.6721, 0.6291, 0.6875], ovf=[0.7172, 0.7173, 0.6617, 0.7237],
      loss=[4.6740, 2.6298, 2.4359, 2.3739, 2.3285, 2.3426, 2.2938, 2.2733, 2.2700, 2.2388,
            2.2608, 2.2254, 2.2311, 2.2154, 2.2283, 2.1835, 2.2607, 2.2092, 2.1888, 2.2255]),
  "teacher RHO batch_interval=600/step_size=100": dict(
      val=[0.6502, 0.6462, 0.5287, 0.6416], ovf=[0.7195, 0.7086, 0.6123, 0.7086],
      loss=[4.4804, 2.7136, 2.5498, 2.4140, 2.4205, 2.3624, 2.4074, 2.3463, 2.3573, 2.3054,
            2.2685, 2.2662, 2.2749, 2.2907, 2.3014, 2.2242, 2.3406, 2.2679, 2.2480, 2.2890]),
}
# same hue per entity as plot_rho_horizon.py: bi200 blue, bi400 orange, bi600 aqua
col = {"400": "#eb6834", "600": "#1baf7a", "200": "#2a78d6"}
rho_ref = {"200": 0.692, "400": 0.653, "600": 0.628}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
cap = ("Actor step_size=100/batch_interval=200, MLP from scratch (no SIL init), 20 epochs, lr=1e-4, seed=42, "
       "MAX_CARDINALITY=2, KEEP_ACTIVE=True, 38 train / 9 val instances; one training run per teacher.")

def style(ax):
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# Fig 1: val + overfit_check (small multiples, shared y)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, key, title in [(axes[0], "val", "Validation (9 VAL_INSTANCES, unseen)"),
                       (axes[1], "ovf", "overfit_check (9 of the train instances)")]:
    for name, r in runs.items():
        t = "400" if "400" in name else "600"
        ax.plot(ep_val, r[key], color=col[t], lw=2, marker="o", ms=7, label=f"Actor, {name}", zorder=3)
        ax.annotate(f"{r[key][-1]:.3f}", (ep_val[-1], r[key][-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    if key == "val":
        for t, v in rho_ref.items():
            ax.axhline(v, color=col[t], lw=1.2, ls="--", zorder=2, label=f"RHO batch_interval={t}/step_size=100 (mean of 3 runs)")
    ax.set_title(title, loc="left", color=INK, fontsize=11); ax.set_xticks(ep_val); ax.set_xlabel("Epoch")
    ax.set_xlim(3.5, 22); style(ax)
axes[0].set_ylabel("Service rate (mean over 9 instances)"); axes[0].set_ylim(0.5, 0.75)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, cap + " Y-axis starts at 0.5.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.86))
for ext in ("png", "pdf"): fig.savefig(out / f"actor_statesync_scratch_val_over_epochs_teacher400_vs_600.{ext}", dpi=200)

# Fig 2: FY loss per epoch
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, r in runs.items():
    t = "400" if "400" in name else "600"
    ax.plot(range(1, 21), r["loss"], color=col[t], lw=2, marker="o", ms=5, label=f"Actor, {name}", zorder=3)
ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("Epoch"); ax.set_ylabel("Mean FY loss (train, per epoch)")
ax.set_title("Training loss over epochs", loc="left", color=INK, fontsize=11); style(ax)
ax.legend(frameon=False, fontsize=9)
fig.text(0.01, 0.01, cap.replace(", MAX_CARD", ",\nMAX_CARD") + " Losses are not directly comparable across teachers (different y* targets).", fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(out / f"actor_statesync_scratch_fyloss_over_epochs_teacher400_vs_600.{ext}", dpi=200)
