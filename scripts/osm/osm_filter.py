"""
2026-11-10: moved here from the session scratchpad (wiped by the OS); unchanged logic from 2026-11-08.
Cuts the drivable road ways inside a bbox out of a Geofabrik PBF and writes them, plus the nodes they
reference, as .osm XML for ox.graph_from_xml. The tag rules replicate OSMnx's network_type="drive" filter
(osmnx/_overpass.py, _get_network_filter), so the result should match what graph_from_polygon would have
downloaded via Overpass (the public Overpass API refused the automated download on 2026-11-08).

Usage: venv_osm/bin/python -I scripts/osm/osm_filter.py <pbf> <out.osm> <west> <south> <east> <north>
"""
import re
import sys
import time

import osmium

pbf, out, w, s, e, n = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7])
BAD_HW = re.compile(r"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|"
                    r"path|pedestrian|planned|platform|proposed|raceway|razed|service|steps|track")
BAD_SVC = re.compile(r"alley|driveway|emergency_access|parking|parking_aisle|private")


def drivable(t):
    hw = t.get("highway")
    if hw is None or BAD_HW.search(hw):
        return False
    if "yes" in t.get("area", "") or "private" in t.get("access", ""):
        return False
    if "no" in t.get("motor_vehicle", "") or "no" in t.get("motorcar", ""):
        return False
    if BAD_SVC.search(t.get("service", "")):
        return False
    return True


t0 = time.time()
kept = 0
# nodes must be read too, otherwise the location cache cannot be filled
fp = osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations()
with osmium.BackReferenceWriter(out, ref_src=pbf, overwrite=True) as writer:
    for way in fp:
        if not way.is_way() or not drivable(way.tags):
            continue
        if any(nd.location.valid() and w <= nd.lon <= e and s <= nd.lat <= n for nd in way.nodes):
            writer.add_way(way)
            kept += 1
print("ways kept", kept, "s", round(time.time() - t0))
