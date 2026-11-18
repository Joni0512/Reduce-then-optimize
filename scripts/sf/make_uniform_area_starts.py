"""
2026-11-16: vehicle starts drawn UNIFORMLY OVER AREA (user idea "Flaeche statt Knoten", go "jo" 2026-11-16).
Copy of scripts/sf/make_uniform_road_starts.py; ONLY the sampling of the start nodes differs.
Why: the uniformroad variant draws uniformly over road NODES, so dense street grids (SF, Oakland) get more vehicles per
km^2. Here every km^2 that contains road has the same weight. Expected (county check 2026-11-16, not yet measured):
area-uniform moves vehicles away from SF (53 % of pickups) towards Marin/San Mateo -> probably LOWER service rate.

Sampling: request counties (same polygon/candidates as uniformroad) -> candidate road nodes projected to EPSG:3310
(California Albers, metres) -> 1 km x 1 km grid cells; only cells containing >= 1 candidate road node count (this
removes open water / roadless land without an extra land mask). Draw n_veh DISTINCT cells uniformly (no two vehicles
in the same cell, analogous to replace=False for nodes), then one road node uniformly inside each drawn cell.
Matrix extension, consistency check and output format identical to uniformroad.

Usage (venv_osm):
  venv_osm/bin/python -I scripts/sf/make_uniform_area_starts.py <base_manifest.json> <seed> [<seed> ...]
Output: <base>_vstart_uniformarea_s<seed>_road_solve.json next to the base manifest.
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point  # noqa: F401  (vectorised via geopandas below)

ROOT = Path(__file__).resolve().parents[2]
GRAPHML = ROOT / "dataset" / "lodes_sf" / "osm" / "sfbay_drive.graphml"
COUNTIES = ROOT / "dataset" / "lodes_sf" / "osm" / "request_counties.geojson"
COUNTY_NAMES = ["San Francisco, California, USA"] + [f"{c} County, California, USA" for c in ("Alameda", "San Mateo", "Marin")]

base_path, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2:]]
t0 = time.time()
m_base = json.loads(base_path.read_text())
M_old = np.asarray(m_base["travel_time_matrix"], dtype=np.float64)
n_old = M_old.shape[0]
n_veh = len(m_base["driver_runs"])
print("base", base_path.name, "matrix", M_old.shape, "vehicles", n_veh, flush=True)

# --- graph as sparse matrix (identical to osm_matrix.py) ---
G = ox.load_graphml(GRAPHML)
nodes = list(G.nodes)
idx = {v: i for i, v in enumerate(nodes)}
u, v, w = [], [], []
for a, b, d in G.edges(data=True):
    u.append(idx[a])
    v.append(idx[b])
    w.append(d["travel_time"])
df = pd.DataFrame({"u": u, "v": v, "w": w}).groupby(["u", "v"]).w.min().reset_index()
A = csr_matrix((df.w.values, (df.u.values, df.v.values)), shape=(len(nodes), len(nodes)))
AT = A.T.tocsr()
print("graph nodes", len(nodes), "s", round(time.time() - t0), flush=True)

# --- existing matrix points -> graph nodes (manifest node_id order: 0 depot, 1+2i pickup, 2+2i dropoff) ---
lat, lon = np.empty(n_old), np.empty(n_old)
lat[0], lon[0] = m_base["depot"]["pt"]["lat"], m_base["depot"]["pt"]["lon"]
for r in m_base["requests"]:
    for k in ("pickup_pt", "dropoff_pt"):
        p = r[k]
        lat[int(p["node_id"])], lon[int(p["node_id"])] = p["lat"], p["lon"]
old_gnode = np.array([idx[o] for o in ox.distance.nearest_nodes(G, lon, lat)])

# --- consistency check: recompute 25 rows of the existing matrix ---
rng_chk = np.random.default_rng(0)
rows = rng_chk.choice(n_old, 25, replace=False)
D = dijkstra(A, directed=True, indices=old_gnode[rows])[:, old_gnode].astype(np.float32)
diff = np.abs(D - M_old[rows])
dev = diff.max()
rel99 = np.percentile(diff / np.maximum(M_old[rows], 1.0), 99)
print(f"consistency check: 25 rows recomputed, max |diff| = {dev:.4f} s, p99 relative diff = {rel99:.2e}", flush=True)
# 2026-11-14: first run with max <= 1 s failed: deviations are systematic but tiny (p99 relative 2.3e-4, max 4.5 s);
# cause unknown (graphml probably not byte-identical to the one the matrix was built from, or rounding). Negligible vs
# minute-scale time windows -> tolerance max 10 s and p99 relative 1e-3; a wrong graph/method would be far off.
assert dev <= 10.0 and rel99 <= 1e-3, "recomputed matrix differs from the manifest matrix -> method/graph mismatch, abort"

# --- candidate start nodes: graph nodes inside the request counties ---
if COUNTIES.exists():
    region = gpd.read_file(COUNTIES).geometry.iloc[0]
else:
    gdf = ox.geocode_to_gdf(COUNTY_NAMES)
    region = gdf.union_all()
    gpd.GeoSeries([region], crs=4326).to_file(COUNTIES, driver="GeoJSON")
    print("geocoded + cached", COUNTIES.relative_to(ROOT), flush=True)
xs = np.array([G.nodes[n]["x"] for n in nodes])
ys = np.array([G.nodes[n]["y"] for n in nodes])
inside = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=4326).within(region).values
cand = np.flatnonzero(inside)
print("candidate road nodes in request counties", len(cand), "of", len(nodes), flush=True)
# 2026-11-16: 1 km grid cells in EPSG:3310 (metres) over the candidate road nodes
_p = gpd.GeoSeries(gpd.points_from_xy(xs[cand], ys[cand]), crs=4326).to_crs(3310)
_cell = np.stack([np.floor(_p.x.values / 1000), np.floor(_p.y.values / 1000)], axis=1).astype(np.int64)
cell_ids, cell_inv = np.unique(_cell, axis=0, return_inverse=True)
cell_inv = cell_inv.ravel()
print("1 km cells with road nodes", len(cell_ids), "(= km^2 of road-covered area)", flush=True)

for seed in seeds:
    # 2026-11-16: area-uniform sampling (see docstring): distinct 1 km cells uniformly, then a node inside each cell
    rng = np.random.default_rng(seed)
    cells_pick = rng.choice(len(cell_ids), size=n_veh, replace=False)
    pick = np.array([rng.choice(cand[cell_inv == c]) for c in cells_pick])
    fwd = dijkstra(A, directed=True, indices=pick)            # from start nodes to everything
    bwd = dijkstra(AT, directed=True, indices=pick)           # from everything to start nodes
    n_new = n_old + n_veh
    M = np.zeros((n_new, n_new), dtype=np.float32)
    M[:n_old, :n_old] = M_old.astype(np.float32)
    M[n_old:, :n_old] = fwd[:, old_gnode]
    M[:n_old, n_old:] = bwd[:, old_gnode].T
    M[n_old:, n_old:] = fwd[:, pick]
    assert np.isfinite(M).all(), "unreachable pair in the extended matrix"
    m = json.loads(base_path.read_text())
    m["travel_time_matrix"] = M.tolist()
    for k, (dr, gi) in enumerate(zip(m["driver_runs"], pick)):
        dr["state"]["loc"] = {"lat": float(ys[gi]), "lon": float(xs[gi]), "node_id": n_old + k}
    m["vehicle_start_info"] = {"mode": "uniform_area_1km_cells_request_counties", "seed": seed, "source": str(base_path),
                               "counties": COUNTY_NAMES, "n_candidates": int(len(cand)), "n_cells": int(len(cell_ids)), "cell_km": 1,
                               "osm_node_ids": [int(nodes[gi]) for gi in pick], "matrix_nodes_added": n_veh}
    out = base_path.with_name(base_path.name.replace("_road_solve.json", f"_vstart_uniformarea_s{seed}_road_solve.json"))
    out.write_text(json.dumps(m))
    tt_dep = M[0, n_old:]
    print(f"seed {seed}: wrote {out.name} ({out.stat().st_size / 1e6:.0f} MB), depot->start tt median "
          f"{np.median(tt_dep) / 60:.1f} min, max {tt_dep.max() / 60:.1f} min, s {round(time.time() - t0)}", flush=True)
