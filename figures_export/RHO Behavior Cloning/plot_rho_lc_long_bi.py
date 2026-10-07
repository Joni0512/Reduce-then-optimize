"""2026-10-06: lc103/lc202 service rate vs batch_interval 10..240 (step_size=10, card 2, KEEP_ACTIVE=True, seed 42).
Sources: bi10: outputs/rho_val_runs/bi10_ss10_card2_keepactiveTrue_seed42/run*/summary.json;
bi20-80: rho_val_bi*_ss10_card2*.log (repo root, 5 repeats + 5 seeds);
bi120-240: outputs/rho_val_runs/bi*_ss10_card2_keepactiveTrue_seed42_lconly/run*/summary.json (outputs/window_analysis/rho_lc_long_bi.py)."""
import json, glob, re, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def vals(bi):
    v = defaultdict(list)
    fs = glob.glob(str(ROOT / f"outputs/rho_val_runs/bi{bi}_ss10_card2_keepactiveTrue_seed42*/run*/summary.json"))
    for f in fs:
        for k, r in json.load(open(f))["rates"].items(): v[k].append(r)
    if not fs:
        for f in glob.glob(str(ROOT / f"rho_val_bi{bi}_ss10_card2*.log")):
            for l in open(f):
                m = re.search(r"\] (\w+): ([0-9.]+) \(", l)
                if m: v[m[1]].append(float(m[2]))
    return v
bis = [10, 20, 40, 60, 80, 120, 160, 240]; data = {b: vals(b) for b in bis}
fig, ax = plt.subplots(figsize=(10, 4.5)); w = 0.1; cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(bis)))
print("inst bi n mean std")
for j, inst in enumerate(["lc103", "lc202"]):
    for k, b in enumerate(bis):
        x = np.array(data[b][inst]); print(inst, b, len(x), round(x.mean(), 3), round(x.std(), 3))
        ax.bar(j + (k - 3.5) * w, x.mean(), w, yerr=x.std(), capsize=2, color=cols[k], label=f"batch_interval={b}" if j == 0 else None)
ax.set_xticks([0, 1], ["lc103\nhorizon 1135, window 1102", "lc202\nhorizon 3289, window 160"])
ax.set_ylim(0.75, 0.97); ax.set_ylabel("service rate (mean ± std over runs)"); ax.grid(axis="y", alpha=.3)
ax.set_title("RHO, step_size=10, cardinality 2: longer lookahead on the clustered instances")
ax.legend(ncol=4, fontsize=8, loc="upper left"); fig.tight_layout()
out = ROOT / "figures_export/RHO Behavior Cloning/rho_lc_long_bi"; fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
