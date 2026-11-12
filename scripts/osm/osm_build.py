"""
2026-11-10: moved here from the session scratchpad (wiped by the OS); unchanged logic from 2026-11-08.
Builds the OSMnx drive graph from the filtered XML, clips it to the region polygon, keeps the largest strongly
connected component and adds free-flow travel times (OSM maxspeed, OSMnx defaults where missing).

Usage: venv_osm/bin/python -I scripts/osm/osm_build.py <filtered.osm> <region.geojson> <out.graphml>
"""
import sys
import time

import geopandas as gpd
import osmnx as ox

xml, region, out = sys.argv[1:4]
t0 = time.time()
G = ox.graph_from_xml(xml, simplify=True, retain_all=True)
print("raw nodes", G.number_of_nodes(), "edges", G.number_of_edges(), "s", round(time.time() - t0), flush=True)
poly = gpd.read_file(region).geometry.iloc[0]
G = ox.truncate.truncate_graph_polygon(G, poly, truncate_by_edge=True)
G = ox.truncate.largest_component(G, strongly=True)
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
print("final nodes", G.number_of_nodes(), "edges", G.number_of_edges(), "s", round(time.time() - t0), flush=True)
ox.save_graphml(G, out)
print("saved s", round(time.time() - t0))
