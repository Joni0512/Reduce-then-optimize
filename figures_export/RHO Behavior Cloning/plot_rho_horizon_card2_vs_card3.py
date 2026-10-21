"""RHO horizon x cardinality on the 9 Li&Lim VAL_INSTANCES (step_size=100, KEEP_ACTIVE=True, seed=42).
card=2: means of runs listed in plot_rho_horizon.py (3 runs for bi200/400/600, 1 for bi800).
card=3: single runs, rho_val_card3_bi{200,400,600,800}_ss100.log (2026-09-28); bi600/bi800 lr208 hit RTV_TIMEOUT=120s once."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__file__).resolve().parent
bis = [200, 400, 600, 800]
card2 = [0.692, 0.653, 0.628, 0.606]
card3 = [0.7712, 0.7407, 0.7343, 0.7277]
C2, C3 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(len(bis)); w = 0.36
for off, vals, c, lab in [(-w/2, card2, C2, "MAX_CARDINALITY=2"), (w/2, card3, C3, "MAX_CARDINALITY=3")]:
    ax.bar(x+off, vals, width=w-0.03, color=c, label=lab, zorder=2)
    for xi, v in zip(x+off, vals):
        ax.text(xi, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, [f"batch_interval={b}\nstep_size=100" for b in bis])
ax.set_ylim(0.5, 0.82); ax.set_ylabel("Service rate (mean over 9 val instances)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("RHO: horizon vs. trip cardinality (validation instances)", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.005, "RHO offline, Li&Lim VAL_INSTANCES (9), KEEP_ACTIVE=True, seed=42. card=2: mean of 3 runs (bi800: 1 run); card=3: 1 run each.\n"
         "Solver non-deterministic (~±1-2 pp between runs). card=3 lr208 at bi600/bi800 hit RTV_TIMEOUT (120 s) once each. Y-axis starts at 0.5.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(out / f"rho_horizon_val_card2_vs_card3_ss100.{ext}", dpi=200)
