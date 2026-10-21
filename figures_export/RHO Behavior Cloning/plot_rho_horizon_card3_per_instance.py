"""RHO card=3 per validation instance, batch_interval=200/400/600/800, step_size=100, KEEP_ACTIVE=True, seed=42.
Single run each, values from rho_val_card3_bi{200,400,600,800}_ss100.log (2026-09-28)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
out = Path(__file__).resolve().parent
inst = ["lc103","lc202","lr103","lr107","lr202","lr208","lrc102","lrc202","lrc205"]
vals = {
 200: [0.9423,0.8431,0.8269,0.8077,0.6000,0.5600,0.7925,0.7647,0.8039],
 400: [0.9231,0.9216,0.8077,0.8077,0.5200,0.6000,0.7925,0.5882,0.7059],
 600: [0.9038,0.8824,0.8269,0.7885,0.5600,0.6000,0.7925,0.5882,0.6667],
 800: [0.8846,0.8627,0.8462,0.8077,0.4800,0.6400,0.7925,0.5490,0.6863],
}
colors = {200:"#2a78d6", 400:"#eb6834", 600:"#1baf7a", 800:"#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":10,"axes.edgecolor":INK2,"xtick.color":INK2,"ytick.color":INK2})
fig, ax = plt.subplots(figsize=(12,5))
xi = np.arange(len(inst)); w = 0.2
for i,(bi,v) in enumerate(vals.items()):
    pos = xi + (i-1.5)*w
    ax.bar(pos, v, width=w-0.02, color=colors[bi], label=f"batch_interval={bi}/step_size=100 (mean {np.mean(v):.3f})", zorder=2)
    for p,y in zip(pos,v):
        ax.text(p, y+0.008, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
ax.set_xticks(xi, inst); ax.set_ylim(0,1.08); ax.set_ylabel("Service rate")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("RHO per validation instance, MAX_CARDINALITY=3", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9, bbox_to_anchor=(1,1.02))
fig.text(0.01,0.005,"RHO offline, Li&Lim VAL_INSTANCES (9), MAX_CARDINALITY=3, KEEP_ACTIVE=True, seed=42, 1 run each (solver non-deterministic, ~±1-2 pp).\n"
         "lr208 at batch_interval=600/800 hit RTV_TIMEOUT (120 s) once each. Source: rho_val_card3_bi*_ss100.log.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0,0.07,1,1))
for ext in ("png","pdf"): fig.savefig(out/f"rho_horizon_val_per_instance_card3_bi200_bi400_bi600_bi800_ss100.{ext}", dpi=200)
