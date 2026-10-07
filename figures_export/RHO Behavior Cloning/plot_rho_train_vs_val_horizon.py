"""2026-10-07: does the RHO horizon pattern from the 9 VAL_INSTANCES also hold on the 38 TRAIN_INSTANCES?
Train: 1 run per config (seed 42, card 2, KEEP_ACTIVE=True), outputs/rho_val_runs/bi*_ss*_card2_keepactiveTrue_seed42_train/run1/summary.json
(rtv_solver/pipeline/rho_val_horizon_runs.py --split train, driver outputs/rho_train_horizon/run_all.sh).
Val: same values as rho_step_vs_batch_sweep_card2_bars (imported from plot_rho_step_vs_batch_sweep_bars.py, which re-saves that figure unchanged)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
here = Path(__file__).resolve().parent; root = here.parent.parent
spec = importlib.util.spec_from_file_location("bars", here / "plot_rho_step_vs_batch_sweep_bars.py")
bars = importlib.util.module_from_spec(spec); spec.loader.exec_module(bars)
val = bars.data
cfgs = {5: [5, 10, 20, 40, 60, 80], 10: [10, 20, 40, 60, 80], 20: [20, 40, 60, 80], 100: [100, 200, 400, 600, 800]}
train = {}
print("step_size batch_interval train_mean n_inst failed val_mean")
for ss, bis in cfgs.items():
    for b in bis:
        d = json.load(open(root / f"outputs/rho_val_runs/bi{b}_ss{ss}_card2_keepactiveTrue_seed42_train/run1/summary.json"))
        train[(ss, b)] = (float(np.mean(list(d["rates"].values()))), len(d["rates"]), list(d["failed"]))
        print(ss, b, round(train[(ss, b)][0], 4), train[(ss, b)][1], train[(ss, b)][2], round(val[ss][b][0], 4))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True, gridspec_kw={"width_ratios": [6, 5, 4, 5]})
w = 0.38
for ax, (ss, bis) in zip(axes, cfgs.items()):
    x = np.arange(len(bis))
    tv = [train[(ss, b)][0] for b in bis]; vv = [val[ss][b][0] for b in bis]
    ax.bar(x - w / 2, vv, w, color="#9db8d9", label="val (9 inst., mean of runs)", zorder=2)
    ax.bar(x + w / 2, tv, w, color="#2a78d6", label="train (38 inst., 1 run)", zorder=2)
    for xi, b, t in zip(x, bis, tv):
        ax.text(xi + w / 2, t + 0.004, f"{t:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK,
                fontweight="bold" if t == max(tv) else None)
        if train[(ss, b)][2]:
            ax.text(xi + w / 2, 0.555, "*", ha="center", color="white", fontsize=11)
    ax.set_xticks(x, [str(b) for b in bis]); ax.set_xlabel("batch_interval")
    ax.set_title(f"step_size={ss}", loc="left", fontsize=11)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.55, 0.92); axes[0].set_ylabel("Service rate"); axes[3].legend(fontsize=8, loc="upper right")
fig.suptitle("RHO horizon pattern: train instances vs val instances", x=0.01, y=0.985, ha="left", fontsize=12)
fig.text(0.01, 0.915, "cardinality 2, KEEP_ACTIVE=True, seed 42; train = 38 TRAIN_INSTANCES, 1 run; "
         "val = 9 VAL_INSTANCES (see rho_step_vs_batch_sweep_card2_bars)", fontsize=8.5, color=INK2)
fig.text(0.01, 0.005, "* = one train instance failed (infeasible ILP), mean over 37. "
         "Sources: outputs/rho_val_runs/*_train/run1/summary.json, plot_rho_step_vs_batch_sweep_bars.py.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(here / f"rho_train_vs_val_horizon.{ext}", dpi=200)
