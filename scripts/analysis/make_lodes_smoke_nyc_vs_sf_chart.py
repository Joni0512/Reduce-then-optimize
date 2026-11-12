"""
2026-11-08: bar chart + table for the first LODES smoke runs (RHO offline, 1000 requests, 50 vehicles),
NYC (haversine 6 m/s, capacity 3) vs SF (OSMnx free-flow road matrix, capacity 2).
Reads final/results.json of both runs; VMT/PMT are summed travel times in seconds (stats_parser.py:333/506).
"""
import glob, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "NYC LODES\nhaversine 6 m/s\ncapacity 3": "outputs/new_tests/nyc_lodes_smoke/bi900_ss450_seed42/run_*/*/final/results.json",
    "SF LODES\nroad free-flow\ncapacity 2": "outputs/outputs/new_tests/sf_lodes_smoke/bi900_ss450_seed42_cap2/run_*/*/final/results.json",
}
HP = "RHO offline | batch_interval=900, step_size=450, max_cardinality=3 | 1000 requests, 50 vehicles, seed 42 | time-rule filter on, no pruner"
OUT = ROOT / "outputs/new_tests/sf_lodes_smoke/lodes_smoke_nyc_vs_sf"

rows = []
for label, pat in RUNS.items():
    path = sorted(glob.glob(str(ROOT / pat)))[-1]
    s = json.load(open(path))["stats"]
    rows.append((label, s, Path(path).relative_to(ROOT)))

metrics = [("Service rate [%]", lambda s: s["serviced"] / 10.0),
           ("Avg. wait [s]", lambda s: s["average_wait_time"]),
           ("Avg. detour [s]", lambda s: s["average_detour"]),
           ("Solver time [s]", lambda s: s["total_time"])]
fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
for ax, (name, f) in zip(axes, metrics):
    vals = [f(s) for _, s, _ in rows]
    bars = ax.bar(range(len(rows)), vals, color=["#4C78A8", "#F58518"])
    ax.bar_label(bars, fmt="%.1f")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_title(name); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(HP, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200)

with open(f"{OUT}.csv", "w") as fh:
    fh.write("run,served,service_rate_pct,avg_wait_s,avg_detour_s,vmt_s,pmt_s,vmt_over_pmt,depot_vmt_s,solver_time_s,source\n")
    for label, s, p in rows:
        fh.write(f"\"{label.replace(chr(10), ' ')}\",{s['serviced']},{s['serviced']/10:.1f},{s['average_wait_time']:.1f},"
                 f"{s['average_detour']:.1f},{s['vmt']:.0f},{s['pmt']:.0f},{s['vmt_over_pmt']:.3f},{s['depot_vmt']:.0f},"
                 f"{s['total_time']:.1f},{p}\n")
print(open(f"{OUT}.csv").read())
