"""
2026-11-11: SF RHO, vehicle capacity 2 vs 3 (user request, see chat). Test instance seed 42, 50 vehicles,
batch_interval=900/step_size=450, max_cardinality=3, keep_active=False, MAX_REQUESTS_PER_VEHICLE=None, solver seed 42,
time-rule filter on/off. Reads final/results.json + config.json (settings from config.json, not folder names):
  capacity 2: outputs/outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2 (filter on),
              outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2_notimerule (filter off)
  capacity 3: outputs/new_tests/sf_lodes_cap3/bi900_ss450_seed42_cap3_{timerule,notimerule}
Capacity is read from the driver_runs of result_driver_runs.json (am_capacity of the vehicles).
Output: figures_export/SF/sf_rho_capacity_2_vs_3.{png,pdf,csv}
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_export" / "SF" / "sf_rho_capacity_2_vs_3"
DIRS = ["outputs/outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2",
        "outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2_notimerule",
        "outputs/new_tests/sf_lodes_cap3/bi900_ss450_seed42_cap3_timerule",
        "outputs/new_tests/sf_lodes_cap3/bi900_ss450_seed42_cap3_notimerule"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

rows = []
for d in DIRS:
    f = Path(glob.glob(str(ROOT / d / "run_*/*/final/results.json"))[0])
    cfg = json.loads((f.parent.parent / "config.json").read_text())["config_dict"]
    s = json.loads(f.read_text())["stats"]
    drv = json.loads((f.parent / "result_driver_runs.json").read_text())["driver_runs"]
    rows.append({"capacity": int(drv[0]["state"]["am_capacity"]), "time_rule_filter": bool(cfg["USE_REQUEST_TIME_RULE_FILTER"]),
                 "batch_interval": cfg["BATCH_INTERVAL"], "step_size": cfg["STEP_SIZE"], "max_cardinality": cfg["MAX_CARDINALITY"],
                 "max_requests_per_vehicle": cfg["MAX_REQUESTS_PER_VEHICLE"], "served": s["serviced"],
                 "service_rate": s["serviced"] / s["total_requests"], "wait_s": round(s["average_wait_time"], 1),
                 "detour_s": round(s["average_detour"], 1), "vmt_over_pmt": round(s["vmt_over_pmt"], 3),
                 "solver_time_s": round(s["total_time"]), "source": str(f.relative_to(ROOT))})

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(2)
for i, (cap, color) in enumerate(((2, BLUE), (3, ORANGE))):
    vals = [next(r for r in rows if r["capacity"] == cap and r["time_rule_filter"] == flt) for flt in (True, False)]
    pos = x + (i - 0.5) * 0.4
    ax.bar(pos, [v["service_rate"] for v in vals], width=0.38, color=color, edgecolor="white", label=f"capacity {cap}")
    for p, v in zip(pos, vals):
        ax.text(p, v["service_rate"] + 0.01, f"{v['service_rate']:.1%}\n({v['served']})", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(x, ["time-rule filter on", "time-rule filter off"])
ax.set_ylim(0, 1)
ax.set_ylabel("Service rate", color=MUTED)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title("RHO on SF seed 42: vehicle capacity 2 vs 3", fontsize=11, loc="left", color=INK)
fig.text(0.01, 0.01, "batch_interval=900/step_size=450, 50 vehicles, max_cardinality=3, keep_active=False, "
         "no per-vehicle request cap, solver seed 42", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)
with open(f"{OUT}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
