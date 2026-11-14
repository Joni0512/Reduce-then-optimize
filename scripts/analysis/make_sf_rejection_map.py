"""
2026-11-12: map of the SF seed 42 pickups: served / rejected but reachable / rejected and unreachable, for the fleet
ablation (user request "1 + 2 + Karte": is the plateau at ~78 % a location problem?).
Classification = scripts/analysis/sf_rho_reachability.py analyse() (same optimistic reachability bound), so the
map and the reachability bar chart cannot disagree.
Background: major roads (motorway/trunk/primary) of dataset/lodes_sf/osm/sfbay_drive.graphml. venv has no osmnx and
venv_osm has no matplotlib, so the road segments are extracted once into a cache:
  ./venv_osm/bin/python3 scripts/analysis/make_sf_rejection_map.py --extract_roads
  ./venv/bin/python3 scripts/analysis/make_sf_rejection_map.py
Output: figures_export/SF/sf_rho_fleet_ablation_map.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRAPHML = ROOT / "dataset" / "lodes_sf" / "osm" / "sfbay_drive.graphml"
ROADS = ROOT / "dataset" / "lodes_sf" / "osm" / "major_roads_segments.npy"
MAJOR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary"}


def extract_roads():
    import networkx as nx
    g = nx.read_graphml(GRAPHML)
    segs = []
    for u, v, d in g.edges(data=True):
        hw = d.get("highway", "")
        if not any(h in hw for h in MAJOR):
            continue
        a, b = g.nodes[u], g.nodes[v]
        segs.append((float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"])))
    np.save(ROADS, np.asarray(segs, dtype=np.float32))
    print(f"wrote {ROADS.relative_to(ROOT)}: {len(segs)} segments")


def make_map():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    sys.path.insert(0, str(Path(__file__).parent))
    import sf_rho_reachability as reach

    fleets = (100, 150, 300)
    runs = [reach.analyse(f"outputs/new_tests/sf_lodes_fleet_ablation/bi900_ss450_mc2_v{v}_notimerule") for v in fleets]
    import glob
    import json
    f = glob.glob(str(ROOT / runs[0]["source"]))[0]
    p = json.loads(Path(f).read_text())
    pick = {r["booking_id"]: (r["pickup_pt"]["lon"], r["pickup_pt"]["lat"]) for r in p["requests"]}
    depot = (p["depot"]["pt"]["lon"], p["depot"]["pt"]["lat"])
    roads = np.load(ROADS) if ROADS.exists() else None

    cats = (("served", "#2a78d6", "served"), ("reach", "#b9b8b2", "rejected, reachable (optimistic bound)"),
            ("unreach", "#eb6834", "rejected, no vehicle could reach it in time"))
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6.6), sharex=True, sharey=True)
    for ax, r in zip(axes, runs):
        if roads is not None:
            ax.add_collection(LineCollection(roads.reshape(-1, 2, 2), colors="#d8d7d2", linewidths=0.4, zorder=0))
        groups = {"served": [], "reach": [], "unreach": []}
        for x in r["rows"]:
            k = "served" if x["served"] else ("unreach" if x["unreachable"] else "reach")
            groups[k].append(pick[x["id"]])
        for k, color, lab in cats:
            pts = np.asarray(groups[k])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, edgecolor="white", linewidth=0.3,
                           label=f"{lab} ({len(pts)})", zorder=2)
        ax.scatter(*depot, marker="*", s=260, color="#0b0b0b", edgecolor="white", linewidth=1, zorder=3, label="depot")
        ax.set_title(f"{r['vehicles']} vehicles", fontsize=11, loc="left", color="#0b0b0b")
        ax.set_aspect(1 / np.cos(np.radians(37.75)))
        ax.tick_params(colors="#52514e", labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=True, framealpha=0.9, fontsize=7.5, loc="upper right")  # 2026-11-12: lower left hid south-peninsula pickups
    lons = np.array([v[0] for v in pick.values()])
    lats = np.array([v[1] for v in pick.values()])
    axes[0].set_xlim(lons.min() - 0.03, lons.max() + 0.03)
    axes[0].set_ylim(lats.min() - 0.03, lats.max() + 0.03)
    fig.suptitle("RHO on SF seed 42: where the requests are served and rejected (pickup locations)", fontsize=12,
                 x=0.01, ha="left", color="#0b0b0b")
    fig.text(0.01, 0.01, "batch_interval=900/step_size=450, time-rule filter off, max_cardinality=2, capacity 2, single depot, "
             "solver seed 42; LRZ cluster. Reachability: nearest vehicle when the request first reaches trip generation, "
             "ignoring its other commitments. Gray lines: major roads (OSM).", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = ROOT / "figures_export" / "SF" / "sf_rho_fleet_ablation_map"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {out}.png/.pdf")


if __name__ == "__main__":
    extract_roads() if "--extract_roads" in sys.argv else make_map()
