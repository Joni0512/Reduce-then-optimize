"""
2026-11-10: moved here from the session scratchpad (wiped by the OS); unchanged logic from 2026-11-08.
Road travel-time matrix (seconds, free-flow) in build_nyc_solve_manifest.py's node order:
0 = depot (mean pickup location), 1+2i = pickup of request i, 2+2i = dropoff (requests sorted by id).
Each point is snapped to its nearest graph node; parallel edges keep the minimum travel time.
Also writes <out>_snap_m.npy with the snap distances in metres.

Usage: venv_osm/bin/python -I scripts/osm/osm_matrix.py <graph.graphml> <requests.csv> <out.npy>
"""
import sys
import time

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

graphml, req_csv, out_npy = sys.argv[1:4]
t0 = time.time()
G = ox.load_graphml(graphml)
print("graph nodes", G.number_of_nodes(), "edges", G.number_of_edges())
c = ["request_id", "pn", "plon", "plat", "dn", "dlon", "dlat", "time"]
r = pd.read_csv(req_csv, header=None, names=c).sort_values("request_id").reset_index(drop=True)
n = len(r)
lat = np.empty(2 * n + 1)
lon = np.empty(2 * n + 1)
lat[0], lon[0] = r.plat.mean(), r.plon.mean()
lat[1::2], lon[1::2] = r.plat.values, r.plon.values
lat[2::2], lon[2::2] = r.dlat.values, r.dlon.values
osm_ids, snap_m = ox.distance.nearest_nodes(G, lon, lat, return_dist=True)
nodes = list(G.nodes)
idx = {v: i for i, v in enumerate(nodes)}
u, v, w = [], [], []
for a, b, d in G.edges(data=True):
    u.append(idx[a])
    v.append(idx[b])
    w.append(d["travel_time"])
# parallel edges: keep the minimum (csr_matrix would sum duplicates)
df = pd.DataFrame({"u": u, "v": v, "w": w}).groupby(["u", "v"]).w.min().reset_index()
A = csr_matrix((df.w.values, (df.u.values, df.v.values)), shape=(len(nodes), len(nodes)))
pt = np.array([idx[o] for o in osm_ids])
uniq, inv = np.unique(pt, return_inverse=True)
print("points", len(pt), "unique graph nodes", len(uniq), "snap m p50/p99/max",
      np.percentile(snap_m, [50, 99, 100]).round(0))
D = dijkstra(A, directed=True, indices=uniq)[:, uniq]
M = D[inv][:, inv].astype(np.float32)
print("inf entries", int(np.isinf(M).sum()), "matrix s", round(time.time() - t0))
np.save(out_npy, M)
np.save(out_npy.replace(".npy", "_snap_m.npy"), snap_m.astype(np.float32))
