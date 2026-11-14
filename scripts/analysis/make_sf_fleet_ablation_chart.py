"""
2026-11-12: SF fleet-size ablation (user request), RHO batch_interval=900/step_size=450, time-rule filter OFF,
max_cardinality=2, am_capacity 2, original single depot, solver seed 42, keep_active=False, test instance seed 42.
Cluster jobs 5732034-38 (submit_sf_lodes_rho_fleet_ablation_bi900_ss450.sbatch, serial_std, 16 CPUs).
Values: if the results were rsync'ed to outputs/new_tests/sf_lodes_fleet_ablation/bi900*/.../final/results.json they are
read from there; otherwise the numbers the user pasted from the cluster (same files, printed 2026-11-12) are used.
Note: stats.vmt is summed travel TIME in seconds (stats_parser.py adds NetworkHandler.travel_time), not distance.
Output: figures_export/SF/sf_rho_fleet_ablation_bi900_ss450_mc2.{png,pdf,csv}
"""
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF" / "sf_rho_fleet_ablation_bi900_ss450_mc2"
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
# pasted from the cluster 2026-11-12: vehicles -> (served, vmt [s], solver time [s])
PASTED = {100: (758, 599_000, 615.9), 150: (790, 677_800, 544.5), 200: (778, 647_400, 693.4),
          250: (782, 653_700, 801.2), 300: (784, 663_100, 969.2)}

rows = []
files = sorted(glob.glob(str(ROOT / "outputs/new_tests/sf_lodes_fleet_ablation/bi900*/**/final/results.json"), recursive=True))
if files:
    for f in files:
        s = json.loads(Path(f).read_text())["stats"]
        nv = int(re.search(r"_v(\d+)_", f).group(1))
        rows.append({"vehicles": nv, "served": s["serviced"], "service_rate": s["serviced"] / s["total_requests"],
                     "wait_s": round(s["average_wait_time"], 1), "detour_s": round(s["average_detour"], 1),
                     "vehicle_travel_time_h": round(s["vmt"] / 3600, 1), "solver_time_s": round(s["total_time"], 1),
                     "source": str(Path(f).relative_to(ROOT))})
else:
    for nv, (srv, vmt, t) in PASTED.items():
        rows.append({"vehicles": nv, "served": srv, "service_rate": srv / 1000, "wait_s": "", "detour_s": "",
                     "vehicle_travel_time_h": round(vmt / 3600, 1), "solver_time_s": t,
                     "source": "cluster outputs/new_tests/sf_lodes_fleet_ablation/bi900_ss450_mc2_v*_notimerule (pasted)"})
rows.sort(key=lambda r: r["vehicles"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(len(rows))
labels = [str(r["vehicles"]) for r in rows]
for ax, key, fmt, ylab, title in (
        (ax1, "service_rate", lambda v: f"{v:.1%}", "Service rate", "Service rate"),
        (ax2, "solver_time_s", lambda v: f"{v:,.0f} s", "Solver time [s]", "Solver time")):
    vals = [r[key] for r in rows]
    ax.bar(x, vals, width=0.6, color=BLUE, edgecolor="white")
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.01, fmt(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Vehicles", color=MUTED)
    ax.set_ylabel(ylab, color=MUTED)
    if key == "service_rate":
        ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
fig.suptitle("RHO on SF seed 42: fleet size", fontsize=11.5, x=0.01, ha="left", color=INK)
fig.text(0.01, 0.01, "batch_interval=900/step_size=450, time-rule filter off, max_cardinality=2, capacity 2, single depot (SF), "
         "keep_active=False, solver seed 42; LRZ cluster serial_std, 16 CPUs", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)
with open(f"{OUT}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
