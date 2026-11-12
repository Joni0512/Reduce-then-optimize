"""
2026-11-11: distributed vehicle start positions for the SF LODES manifests (user request "Erst verteilte Starts", see chat).
Cheap test whether the start position matters before refactoring rebalancing into RHO: all 50 vehicles currently start at
the depot (mean pickup location, in SF), and pickups > 20 min from the depot are almost never served.

No solver change needed: VehicleHandler.add_manifest_to_vehicles (called in every RHO step, online_rtv_solver.py:141)
sets vehicle.last_node / next_immediate_node from driver_run.state.loc, overriding the depot start of _load_vehicles.
This script only rewrites driver_runs[*].state.loc of an existing manifest: vehicle v starts at the pickup node of a
request sampled uniformly without replacement (seeded) -> start positions follow the demand density; every pickup is
already a node of the travel-time matrix, so no new matrix is needed. Depot, requests and matrix stay unchanged.
Caveat: the sampled requests (n_vehicles of 1000) get a vehicle right at their pickup -> small advantage for them;
this is a diagnostic, not a fair method.

2026-11-11: --mode two_depots (user request "2 Depots, eins ausserhalb"): depot A = the manifest's depot (mean pickup,
in SF); depot B = mean lat/lon of all pickups with free-flow travel time > --far_s (default 1200 s = 20 min) from A,
snapped to the nearest pickup node (haversine) so it is a matrix node. Each request is assigned to the depot with the
smaller travel time to its pickup; vehicles are split proportionally to that demand (rounded, >= 1 each). The payload
depot (used only for the depot-return feasibility check) stays A. Output suffix "_vstart_twodepots".

Usage: ./venv/bin/python3 scripts/sf/make_distributed_vehicle_starts.py \
           --in solutions/sf/manifests/sf_lodes1000_0608_seed42_v50_cap2_road_solve.json --seed 42
Output: same path with "_road_solve" -> "_vstart_pickups_road_solve".
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["random_pickups", "two_depots"], default="random_pickups")  # 2026-11-11
    p.add_argument("--far_s", type=float, default=1200.0)  # 2026-11-11: two_depots, "far" threshold from depot A
    a = p.parse_args()
    if a.mode == "two_depots":
        return two_depots(a)

    src = ROOT / a.inp
    m = json.loads(src.read_text())
    reqs, runs = m["requests"], m["driver_runs"]
    assert len(runs) <= len(reqs)
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(reqs), size=len(runs), replace=False)
    for dr, i in zip(runs, pick):
        pt = reqs[int(i)]["pickup_pt"]
        dr["state"]["loc"] = {"lat": pt["lat"], "lon": pt["lon"], "node_id": int(pt["node_id"])}
    m["vehicle_start_info"] = {"mode": "random_pickup_nodes", "seed": a.seed, "source": str(src.relative_to(ROOT)),
                               "booking_ids": [int(reqs[int(i)]["booking_id"]) for i in pick]}
    out = src.with_name(src.name.replace("_road_solve", "_vstart_pickups_road_solve"))
    assert out != src
    out.write_text(json.dumps(m))
    tt = np.asarray(m["travel_time_matrix"])
    d = tt[int(m["depot"]["node_id"]), [int(reqs[int(i)]["pickup_pt"]["node_id"]) for i in pick]] / 60
    print(f"wrote {out.relative_to(ROOT)}: {len(runs)} vehicles, depot->start free-flow min/median/max "
          f"{d.min():.1f}/{np.median(d):.1f}/{d.max():.1f} min")


def two_depots(a):
    """2026-11-11: see module docstring (--mode two_depots)."""
    src = ROOT / a.inp
    m = json.loads(src.read_text())
    reqs, runs = m["requests"], m["driver_runs"]
    tt = np.asarray(m["travel_time_matrix"], dtype=float)
    a_node = int(m["depot"]["node_id"])
    pk = np.array([int(r["pickup_pt"]["node_id"]) for r in reqs])
    lat = np.array([r["pickup_pt"]["lat"] for r in reqs])
    lon = np.array([r["pickup_pt"]["lon"] for r in reqs])
    far = tt[a_node, pk] > a.far_s
    clat, clon = lat[far].mean(), lon[far].mean()
    dist = np.hypot((lat - clat) * 111_000, (lon - clon) * 111_000 * np.cos(np.radians(clat)))
    j = int(dist.argmin())
    b_node = int(pk[j])
    to_b = tt[b_node, pk] < tt[a_node, pk]
    n_b = int(np.clip(round(len(runs) * to_b.mean()), 1, len(runs) - 1))
    a_loc = {"lat": m["depot"]["pt"]["lat"], "lon": m["depot"]["pt"]["lon"], "node_id": a_node}
    b_loc = {"lat": float(lat[j]), "lon": float(lon[j]), "node_id": b_node}
    for v, dr in enumerate(runs):
        dr["state"]["loc"] = dict(b_loc if v < n_b else a_loc)
    m["vehicle_start_info"] = {"mode": "two_depots", "far_s": a.far_s, "depot_A": a_loc, "depot_B": b_loc,
                               "depot_B_snapped_from": [float(clat), float(clon)], "vehicles_A": len(runs) - n_b,
                               "vehicles_B": n_b, "source": str(src.relative_to(ROOT))}
    out = src.with_name(src.name.replace("_road_solve", "_vstart_twodepots_road_solve"))
    assert out != src
    out.write_text(json.dumps(m))
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"far pickups (> {a.far_s:.0f} s from A): {int(far.sum())}; B centroid ({clat:.4f}, {clon:.4f}) snapped "
          f"{dist[j]:.0f} m to pickup node {b_node} ({lat[j]:.4f}, {lon[j]:.4f})")
    print(f"A ({a_loc['lat']:.4f}, {a_loc['lon']:.4f}); travel time A->B {tt[a_node, b_node]/60:.1f} min, B->A {tt[b_node, a_node]/60:.1f} min")
    print(f"requests closer to B: {int(to_b.sum())} of {len(reqs)} -> vehicles A={len(runs) - n_b}, B={n_b}")
    near = np.minimum(tt[a_node, pk], tt[b_node, pk]) / 60
    print(f"pickups > 20 min from nearest depot: A only {int((tt[a_node, pk] > 1200).sum())}, A or B {int((near > 20).sum())}")


if __name__ == "__main__":
    main()
