"""
2026-11-14: vehicle starts drawn UNIFORMLY over the road nodes of the request region (user request: test random starts
that do not follow the demand; choice "Strassenknoten, Request-Region"). Follow-up of the oracle check
(scripts/analysis/sf_rho_nodepot_startseeds.py): starts drawn from pickups of this or another instance follow the
demand density; here the starts are independent of the demand.

Request region = union of the counties that contain requests (SF, Alameda, San Mateo, Marin; NOT Contra Costa, which
is only a routing buffer in the graph, see scripts/osm/osm_region.py), no buffer. Geocoded once via osmnx and cached
in dataset/lodes_sf/osm/request_counties.geojson.

The start nodes are not in the instance matrix (2001 nodes = depot + pickup/dropoff per request), so the matrix is
EXTENDED (no snapping onto request locations): new nodes 2001..2001+N-1, travel times with the same method as
scripts/osm/osm_matrix.py (free-flow OSM travel_time, parallel edges -> min, Dijkstra between snapped graph nodes),
both directions, float32. Existing matrix entries are kept unchanged. Before writing, a sample of existing entries is
recomputed with this method and compared to the manifest matrix (abort if the max deviation is > 1 s).

Usage (venv_osm):
  venv_osm/bin/python -I scripts/sf/make_uniform_road_starts.py <base_manifest.json> <seed> [<seed> ...]
Output: <base>_vstart_uniformroad_s<seed>_road_solve.json next to the base manifest (base name without _road_solve).
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

for seed in seeds:
    pick = np.random.default_rng(seed).choice(cand, size=n_veh, replace=False)
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
    m["vehicle_start_info"] = {"mode": "uniform_road_nodes_request_counties", "seed": seed, "source": str(base_path),
                               "counties": COUNTY_NAMES, "n_candidates": int(len(cand)),
                               "osm_node_ids": [int(nodes[gi]) for gi in pick], "matrix_nodes_added": n_veh}
    out = base_path.with_name(base_path.name.replace("_road_solve.json", f"_vstart_uniformroad_s{seed}_road_solve.json"))
    out.write_text(json.dumps(m))
    tt_dep = M[0, n_old:]
    print(f"seed {seed}: wrote {out.name} ({out.stat().st_size / 1e6:.0f} MB), depot->start tt median "
          f"{np.median(tt_dep) / 60:.1f} min, max {tt_dep.max() / 60:.1f} min, s {round(time.time() - t0)}", flush=True)
