"""
Builds a synthetic NYC commuter request file from Census LODES8 data, in the same
headerless column layout as RollingHorizon's data/requests/requests.csv:
request_id, pickup_node, pickup_lon, pickup_lat, dropoff_node, dropoff_lon, dropoff_lat, time

2026-11-06: new script. Adapted from the idea in DMadhuranga/scripts-for-simulator
(generate_data.py), deliberately not a copy of it:
- reuses the existing NYC map (nodes.csv) instead of building an OSMnx graph
- departure times are drawn independently of the OD pair; the original assigns them
  by row order after sorting by job count, which clusters identical OD pairs in time
- samples n jobs with a seed; the original emits one request per job
- morning direction only; the original also adds a mirrored PM trip per job

LODES has no departure times and no travel mode: the time-of-day profile here is an
assumption, and each output file is a synthetic instance, not a real calendar day.
od_main only covers workers living and working in NY state.

Usage:
    python3 -m rtv_solver.pipeline.build_nyc_lodes_requests \
        --od-csv ~/Downloads/ny_od_main_JT00_2023.csv \
        --block-dbf ~/Downloads/tl_2020_36_tabblock20/tl_2020_36_tabblock20.dbf \
        --n-requests 1000 --seed 42
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")

LAT_M = 110574.0
LON_M = np.cos(np.radians(40.75)) * 111320.0


def read_block_points(dbf_path: Path) -> pd.DataFrame:
    # 2026-11-06: plain DBF parsing so the TIGER block table can be read without adding
    # geopandas to the venv; the interior points are all this script needs because
    # every location is snapped to a map node anyway.
    with open(dbf_path, "rb") as f:
        header = f.read(32)
        n_rows = struct.unpack("<I", header[4:8])[0]
        header_len, record_len = struct.unpack("<HH", header[8:12])
        fields = {}
        offset = 1  # byte 0 of each record is the deletion flag
        while True:
            descriptor = f.read(32)
            if descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\0")[0].decode("latin-1")
            fields[name] = (offset, descriptor[16])
            offset += descriptor[16]
        f.seek(header_len)
        raw = f.read(n_rows * record_len)

    def column(name):
        start, length = fields[name]
        return [
            raw[i * record_len + start: i * record_len + start + length].decode("latin-1").strip()
            for i in range(n_rows)
        ]

    return pd.DataFrame({
        "geoid": pd.Series(column("GEOID20")).astype("int64"),
        "lat": pd.Series(column("INTPTLAT20")).astype(float),
        "lon": pd.Series(column("INTPTLON20")).astype(float),
    }).set_index("geoid")


def snap_to_nodes(blocks: pd.DataFrame, nodes: pd.DataFrame, chunk: int = 2000) -> pd.DataFrame:
    """Returns the nearest map node and its distance for each block (equirectangular metres)."""
    node_x, node_y = nodes["lon"].values * LON_M, nodes["lat"].values * LAT_M
    x, y = blocks["lon"].values * LON_M, blocks["lat"].values * LAT_M
    nearest = np.empty(len(blocks), dtype=np.int64)
    dist = np.empty(len(blocks))
    for i in range(0, len(blocks), chunk):
        d2 = (x[i:i + chunk, None] - node_x) ** 2 + (y[i:i + chunk, None] - node_y) ** 2
        j = d2.argmin(axis=1)
        nearest[i:i + chunk] = nodes["node_id"].values[j]
        dist[i:i + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    out = blocks.copy()
    out["node"] = nearest
    out["snap_m"] = dist
    return out


def matches_geoid_prefix(geoids, prefixes) -> np.ndarray:
    # 2026-11-08: region selection for cities without a prepared node map (San Francisco):
    # a 15-digit block GEOID starts with state(2)+county(3)+tract(6), so a prefix of
    # length L matches when geoid // 10**(15-L) equals the prefix.
    geoids = np.asarray(geoids, dtype=np.int64)
    hit = np.zeros(len(geoids), dtype=bool)
    for p in prefixes:
        hit |= (geoids // 10 ** (15 - len(p))) == int(p)
    return hit


def seconds_to_hms(t: int) -> str:
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def hms_to_seconds(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--od-csv", required=True, type=Path)
    parser.add_argument("--block-dbf", required=True, type=Path)
    parser.add_argument("--nodes-csv", default=str(RH_ROOT / "data/map/nodes.csv"),
                        help="node map to snap blocks to; 'none' keeps block interior points and "
                             "writes a running block index into the node columns (2026-11-08)")
    # 2026-11-08: region filters for San Francisco (work in SF county, home in SF/Alameda/
    # San Mateo/Marin, agreed with the user); unset means no filter, so the NYC path is unchanged.
    parser.add_argument("--work-geoid-prefixes", default="",
                        help="comma list of GEOID prefixes the workplace block must match, e.g. 06075")
    parser.add_argument("--home-geoid-prefixes", default="",
                        help="comma list of GEOID prefixes the home block must match, e.g. 06075,06001")
    parser.add_argument("--exclude-geoid-prefixes", default="",
                        help="comma list of GEOID prefixes to drop for both ends, e.g. island tracts")
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snap-radius-m", type=float, default=500.0,
                        help="drop blocks further than this from the nearest map node")
    parser.add_argument("--start", default="06:00:00")
    parser.add_argument("--end", default="08:00:00")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: dataset/lodes_nyc/requests_lodes<n>_<start>_<end>_seed<seed>.csv")
    args = parser.parse_args()

    split = lambda s: [p.strip() for p in s.split(",") if p.strip()]
    work_prefixes = split(args.work_geoid_prefixes)
    home_prefixes = split(args.home_geoid_prefixes)
    exclude_prefixes = split(args.exclude_geoid_prefixes)
    use_map = args.nodes_csv.lower() != "none"

    blocks = read_block_points(args.block_dbf.expanduser())
    # 2026-11-08: block-level region filters, applied before snapping; all no-ops when unset.
    if work_prefixes or home_prefixes:
        blocks = blocks[matches_geoid_prefix(blocks.index, work_prefixes + home_prefixes)]
    if exclude_prefixes:
        blocks = blocks[~matches_geoid_prefix(blocks.index, exclude_prefixes)]

    if use_map:
        nodes = pd.read_csv(args.nodes_csv, header=None, names=["node_id", "lat", "lon"])
        margin = 0.01  # degrees, roughly 1 km; only a speed-up before the exact distance check
        in_box = (
            blocks["lat"].between(nodes["lat"].min() - margin, nodes["lat"].max() + margin)
            & blocks["lon"].between(nodes["lon"].min() - margin, nodes["lon"].max() + margin)
        )
        snapped = snap_to_nodes(blocks[in_box], nodes)
        snapped = snapped[snapped["snap_m"] <= args.snap_radius_m]
    else:
        # 2026-11-08: no node map for SF; each block is its own location, so the
        # same-node removal below becomes a same-block removal.
        snapped = blocks.copy()
        snapped["node"] = np.arange(1, len(snapped) + 1)
        snapped["snap_m"] = 0.0

    od = pd.read_csv(args.od_csv.expanduser(), usecols=["w_geocode", "h_geocode", "S000"], dtype="int64")
    total_jobs = int(od["S000"].sum())
    od = od[od["h_geocode"].isin(snapped.index) & od["w_geocode"].isin(snapped.index)]
    # 2026-11-08: direction-specific filters (workplace in SF, home in the commute region).
    if work_prefixes:
        od = od[matches_geoid_prefix(od["w_geocode"].values, work_prefixes)]
    if home_prefixes:
        od = od[matches_geoid_prefix(od["h_geocode"].values, home_prefixes)]
    jobs_on_map = int(od["S000"].sum())
    home = snapped.loc[od["h_geocode"].values]
    work = snapped.loc[od["w_geocode"].values]
    # A request whose pickup and dropoff share a node has zero travel time.
    keep = home["node"].values != work["node"].values
    od, home, work = od[keep], home[keep], work[keep]
    jobs_usable = int(od["S000"].sum())
    if args.n_requests > jobs_usable:
        parser.error(f"--n-requests {args.n_requests} exceeds the {jobs_usable} usable jobs")

    # Each job is one unit, so OD pairs are drawn in proportion to S000, without replacement.
    rng = np.random.default_rng(args.seed)
    job_rows = np.repeat(np.arange(len(od)), od["S000"].values)
    chosen = rng.choice(job_rows, size=args.n_requests, replace=False)

    start, end = hms_to_seconds(args.start), hms_to_seconds(args.end)
    # Drawn independently of the OD row and sorted afterwards; ids follow departure time.
    times = np.sort(rng.integers(start, end, size=args.n_requests))
    rng.shuffle(chosen)

    out_df = pd.DataFrame({
        "request_id": np.arange(args.n_requests),
        "pickup_node": home["node"].values[chosen],
        "pickup_lon": home["lon"].values[chosen],
        "pickup_lat": home["lat"].values[chosen],
        "dropoff_node": work["node"].values[chosen],
        "dropoff_lon": work["lon"].values[chosen],
        "dropoff_lat": work["lat"].values[chosen],
        "time": [seconds_to_hms(int(t)) for t in times],
    })

    if args.out is None:
        tag = f"{args.start[:2]}{args.end[:2]}"
        args.out = ROOT / f"dataset/lodes_nyc/requests_lodes{args.n_requests}_{tag}_seed{args.seed}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, header=False, index=False)

    meta = {
        "od_csv": str(args.od_csv), "block_dbf": str(args.block_dbf), "nodes_csv": str(args.nodes_csv),
        "n_requests": args.n_requests, "seed": args.seed, "snap_radius_m": args.snap_radius_m,
        "start": args.start, "end": args.end,
        "work_geoid_prefixes": work_prefixes, "home_geoid_prefixes": home_prefixes,
        "exclude_geoid_prefixes": exclude_prefixes,
        "jobs_total": total_jobs, "jobs_on_map": jobs_on_map, "jobs_usable": jobs_usable,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(out_df)} requests -> {args.out}")
    print(f"Jobs: {total_jobs} total, {jobs_on_map} on map, {jobs_usable} usable after same-node removal")


if __name__ == "__main__":
    main()
