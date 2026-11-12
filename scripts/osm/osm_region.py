"""
2026-11-10: moved here from the session scratchpad (wiped by the OS); unchanged logic from 2026-11-08.
Geocodes the SF commute region (SF + Alameda, San Mateo, Marin; Contra Costa only as routing buffer so
East Bay <-> Marin trips can use the Richmond-San Rafael bridge), buffers 2 km and saves the polygon as
GeoJSON; prints its bbox for osm_filter.py.

Usage: venv_osm/bin/python -I scripts/osm/osm_region.py dataset/lodes_sf/osm/region.geojson
"""
import sys

import geopandas as gpd
import osmnx as ox

ox.settings.use_cache = False
names = ["San Francisco, California, USA"] + [
    f"{c} County, California, USA" for c in ("Alameda", "San Mateo", "Marin", "Contra Costa")
]
gdf = ox.geocode_to_gdf(names)
print(gdf[["display_name"]].to_string())
poly = gpd.GeoSeries([gdf.to_crs(3310).union_all().buffer(2000)], crs=3310).to_crs(4326)
poly.to_file(sys.argv[1], driver="GeoJSON")
print("bbox", *[round(v, 4) for v in poly.total_bounds])
