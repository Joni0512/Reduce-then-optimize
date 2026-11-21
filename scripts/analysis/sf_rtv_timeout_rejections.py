"""
2026-11-18: which requests are rejected at batch_interval=3600/step_size=1800 with rtv_timeout 1800 s vs 21600 s
(user: "warum ist rtv 21600 schlechter" -> "pruefe welche Requests abgelehnt werden"). Read-only analysis.
Input: RHO baselines on SF test seed 42, 150 vehicles, uniform road starts (start seeds 42-46), max_cardinality=2,
time-rule filter off, no depot return; final/results.json (stats.serviced_requests) of
  outputs/new_tests/sf_lodes_nodepot/bi3600_ss1800_mc2_v150_notimerule_noreturn_uniformroad{,_rtv21600}_s{SS}/
and the request attributes from solutions/sf/manifests/sf_lodes1000_0608_seed42_v150_cap2_vstart_uniformroad_s{SS}_road_solve.json.
Per request: pickup window start (-> decision step: first decided at t = floor(start/1800)*1800 - 1800 or 0), position in the
first/second half of its first window, direct travel time (matrix pickup->dropoff), distance of the pickup to the nearest
vehicle start. Output: outputs/rtv_timeout_check/rejections_by_request.csv + printed summary.
"""
import csv, json, glob, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/new_tests/sf_lodes_nodepot"
OUT = ROOT / "outputs/rtv_timeout_check"
STEP, BI = 1800, 3600


def served(variant, ss):
    f = glob.glob(str(D / f"bi3600_ss1800_mc2_v150_notimerule_noreturn_uniformroad{variant}_s{ss}/run_*/*/final/results.json"))
    assert len(f) == 1, f
    return {int(x) for x in json.load(open(f[0]))["stats"]["serviced_requests"]}


rows = []
for ss in (42, 43, 44, 45, 46):
    man = json.load(open(ROOT / f"solutions/sf/manifests/sf_lodes1000_0608_seed42_v150_cap2_vstart_uniformroad_s{ss}_road_solve.json"))
    tt = np.asarray(man["travel_time_matrix"], dtype=float) if not isinstance(man["travel_time_matrix"], str) else None
    starts = [dr["state"]["loc"]["node_id"] if isinstance(dr.get("state", {}).get("loc"), dict) else None for dr in man["driver_runs"]]
    a, b = served("", ss), served("_rtv21600", ss)
    for r in man["requests"]:
        i, p, q = r["booking_id"], r["pickup_pt"]["node_id"], r["dropoff_pt"]["node_id"]
        s0 = r["pickup_time_window_start"]
        rows.append({"start_seed": ss, "request": i, "pickup_start": s0, "decision_step": s0 // STEP,
                     "direct_tt": tt[p, q] if tt is not None else "", "nearest_start_tt": min(tt[v, p] for v in starts if v is not None) if tt is not None and any(starts) else "",
                     "served_rtv1800": int(i in a), "served_rtv21600": int(i in b)})
with open(OUT / "rejections_by_request.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def grp(r):
    return {(1, 1): "both served", (1, 0): "only rtv1800 served", (0, 1): "only rtv21600 served", (0, 0): "both rejected"}[(r["served_rtv1800"], r["served_rtv21600"])]
print("per start seed: served rtv1800 / rtv21600 / only1800 / only21600")
for ss in (42, 43, 44, 45, 46):
    R = [r for r in rows if r["start_seed"] == ss]
    print(ss, sum(r["served_rtv1800"] for r in R), sum(r["served_rtv21600"] for r in R),
          sum(grp(r) == "only rtv1800 served" for r in R), sum(grp(r) == "only rtv21600 served" for r in R))
print("\nrejected per pickup hour (pickup_start // 3600), all start seeds pooled: rtv1800 / rtv21600 / requests")
for h in sorted({r["pickup_start"] // 3600 for r in rows}):
    R = [r for r in rows if r["pickup_start"] // 3600 == h]
    print(f"hour {h}: {sum(1 - r['served_rtv1800'] for r in R)} / {sum(1 - r['served_rtv21600'] for r in R)} / {len(R)}")
print("\nrejected per 30-min slot (pickup_start // 1800): rtv1800 / rtv21600 / requests")
for h in sorted({r["pickup_start"] // 1800 for r in rows}):
    R = [r for r in rows if r["pickup_start"] // 1800 == h]
    print(f"slot {h} [{h*1800}-{h*1800+1800}): {sum(1 - r['served_rtv1800'] for r in R)} / {sum(1 - r['served_rtv21600'] for r in R)} / {len(R)}")
print("\ngroup: n, median direct_tt, median nearest_vehicle_start_tt, median pickup_start")
for g in ("both served", "only rtv1800 served", "only rtv21600 served", "both rejected"):
    R = [r for r in rows if grp(r) == g]
    if R and R[0]["direct_tt"] != "":
        print(g, len(R), round(st.median(r["direct_tt"] for r in R)), round(st.median(r["nearest_start_tt"] for r in R)) if R[0]["nearest_start_tt"] != "" else "-", round(st.median(r["pickup_start"] for r in R)))
