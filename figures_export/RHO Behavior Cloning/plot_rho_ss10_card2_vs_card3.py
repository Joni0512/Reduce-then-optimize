"""2026-10-04: RHO step_size=10, card 2 vs card 3, 9 VAL_INSTANCES, KEEP_ACTIVE=True.
card 2: 10 runs/config (5 repeats seed 42 + seeds 1-5, local); card 3: 5 runs/config (LRZ serial_std, seed 42).
Service rate = mean of per-instance means over the 8 instances with >=1 successful run in every config (lrc202
excluded: card 3 bi20 aborted 5/5). Source: outputs/rho_card3_ss10/summary.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parent.parent
S = json.load(open(root / "outputs/rho_card3_ss10/summary.json")); bis = [20, 40, 60, 80]
runs = {"card2": 10, "card3": 5}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; C = {"card2": "#2a78d6", "card3": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
x = np.arange(4); w = 0.36
for off, n, lab in [(-w/2, "card2", "MAX_CARDINALITY=2 (10 runs)"), (w/2, "card3", "MAX_CARDINALITY=3 (5 runs)")]:
    m = [S["means"][f"{n}_bi{b}"] for b in bis]
    a1.bar(x + off, m, w - 0.04, color=C[n], label=lab, zorder=2)
    for xi, v in zip(x + off, m): a1.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ab = [100 * sum(S["fails"][f"{n}_bi{b}"].values()) / (9 * runs[n]) for b in bis]
    a2.bar(x + off, ab, w - 0.04, color=C[n], label=lab, zorder=2)
    for xi, v in zip(x + off, ab): a2.text(xi, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
for ax in (a1, a2):
    ax.set_xticks(x, [f"batch_interval\n={b}" for b in bis], fontsize=9)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
a1.set_ylim(0.80, 0.95); a1.set_ylabel("Service rate (8 common val instances)")
a1.set_title("RHO service rate (y-axis starts at 0.80)", loc="left", fontsize=11, color=INK); a1.legend(frameon=False, fontsize=9, loc="upper left")
a2.set_ylabel("Instance runs aborted (infeasible ILP) [%]"); a2.set_title("RHO infeasible-ILP abort rate", loc="left", fontsize=11, color=INK)
fig.text(0.01, 0.005, "RHO offline, step_size=10, KEEP_ACTIVE=True, 9 Li&Lim VAL_INSTANCES. Service rate = mean of per-instance means over successful runs; lrc202 excluded (card 3 bi20: 5/5 aborted).\n"
         "Caveat: card-3 means of lr202/lr208 rest on 1-3 surviving runs (survivorship bias). Sources: rho_val_bi*_ss10_card2*.log, outputs/rho_card3_ss10/*.log.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"): fig.savefig(here / f"rho_ss10_card2_vs_card3_service_and_aborts.{ext}", dpi=200)
