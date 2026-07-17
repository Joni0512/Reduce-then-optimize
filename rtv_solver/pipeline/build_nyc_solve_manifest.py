"""
Builds a *solve-ready* NYC manifest for rtv_solver/main.py (--mode offline/coaml),
as opposed to build_nyc_manifest.py's output which only exists to carry expert
pair labels for GNN training (its driver_runs are pre-assigned 2-stop "pairs",
which PayloadParser would treat as already-boarded/active requests - wrong for
an actual fresh solve).

Here, driver_runs are empty vehicles (canonical state shape, manifest=[]) so
every request starts unassigned and the solver has to build trips/routes from
scratch - matching how solutions/li_lim/manifests/*.json are structured.

Usage:
    python3 -m rtv_solver.pipeline.build_nyc_solve_manifest \
        --requests-csv data/requests/requests_dense2000_30min.csv \
        --n-requests 150 --n-vehicles 75 --seed 42 \
        --out solutions/nyc/manifests/nyc_pilot150_solve.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")

MAX_WAITING = 300
MAX_DETOUR = 600
ASSUMED_SPEED_MPS = 6.0
VEHICLE_END_TIME_BUFFER = 1200  # extra seconds past the last request's dropoff window end


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_time_to_seconds(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-csv", default="data/requests/requests_dense2000_30min.csv",
                         help="path (relative to RollingHorizon root) of the request slice to sample from")
    parser.add_argument("--n-requests", type=int, default=150)
    parser.add_argument("--n-vehicles", type=int, default=75)
    parser.add_argument("--am-capacity", type=int, default=3, help="matches CARSIZE used for the NYC baseline runs")
    parser.add_argument("--wc-capacity", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="solutions/nyc/manifests/nyc_pilot150_solve.json",
                         help="output path relative to this (Reduce-then-optimize) repo root")
    # 2026-07-16: restrict sampling to a holdout ID list so pilots don't overlap with
    # whatever request subset the GNN checkpoint was trained on - see
    # nyc_dense2000_holdout_ids.csv / build_nyc_manifest.py's --request-ids-file.
    parser.add_argument("--request-ids-file", type=str, default=None,
                         help="optional CSV with a 'request_id' column - if set, sample only "
                              "from these IDs (e.g. the holdout split, never seen in training).")
    args = parser.parse_args()

    requests_csv = RH_ROOT / args.requests_csv
    reqs_all = pd.read_csv(
        requests_csv, header=None,
        names=["request_id", "pickup_node", "pickup_lon", "pickup_lat",
               "dropoff_node", "dropoff_lon", "dropoff_lat", "time"],
    )
    if args.request_ids_file:
        allowed_ids = set(pd.read_csv(ROOT / args.request_ids_file)["request_id"])
        reqs_all = reqs_all[reqs_all["request_id"].isin(allowed_ids)].reset_index(drop=True)
    reqs = reqs_all.sample(n=args.n_requests, random_state=args.seed).sort_values("request_id").reset_index(drop=True)

    reqs["t_sec"] = reqs["time"].apply(parse_time_to_seconds)
    reqs["t_sec"] -= reqs["t_sec"].min()

    travel_time_s = haversine_m(
        reqs["pickup_lat"], reqs["pickup_lon"], reqs["dropoff_lat"], reqs["dropoff_lon"]
    ) / ASSUMED_SPEED_MPS

    requests_out = []
    max_dropoff_window_end = 0
    for i, row in reqs.iterrows():
        pickup_start = int(row["t_sec"])
        pickup_end = pickup_start + MAX_WAITING
        dropoff_start = pickup_start + int(travel_time_s[i])
        dropoff_end = dropoff_start + MAX_DETOUR
        max_dropoff_window_end = max(max_dropoff_window_end, dropoff_end)

        requests_out.append({
            "booking_id": int(row["request_id"]),
            "pickup_pt": {"lat": row["pickup_lat"], "lon": row["pickup_lon"], "node_id": 1 + 2 * i},
            "dropoff_pt": {"lat": row["dropoff_lat"], "lon": row["dropoff_lon"], "node_id": 2 + 2 * i},
            "pickup_time_window_start": pickup_start,
            "pickup_time_window_end": pickup_end,
            "dropoff_time_window_start": dropoff_start,
            "dropoff_time_window_end": dropoff_end,
            "am": 1,
            "wc": 0,
            "pickup_service_time": 90,
            "dropoff_service_time": 90,
        })

    depot = {
        "pt": {"lat": float(reqs["pickup_lat"].mean()), "lon": float(reqs["pickup_lon"].mean())},
        "node_id": 0,
    }

    vehicle_end_time = max_dropoff_window_end + VEHICLE_END_TIME_BUFFER

    # Canonical empty-vehicle state shape, matching PayloadParser.normalize_to_canonical's
    # chattanooga->wilson conversion template (payload_parser.py ~line 416) - "manifest": []
    # is required (not omitted) so PayloadKeys.DRIVER_MANIFEST-in-driver_run checks stay
    # well-defined, but an empty list means no pre-assigned/boarded requests.
    driver_runs = [
        {
            "state": {
                "run_id": v,
                "start_time": 0,
                "end_time": vehicle_end_time,
                "am_capacity": args.am_capacity,
                "wc_capacity": args.wc_capacity,
                "locations_already_serviced": 0,
                "location_dt_seconds": 0,
                # node_id must match the depot's own node_id (0) - Node.from_dict()
                # leaves it None otherwise, which crashes travel_time_from_matrix's
                # int(source.node_id * no_of_nodes + dest.node_id) indexing.
                "loc": {"lat": depot["pt"]["lat"], "lon": depot["pt"]["lon"], "node_id": 0},
            },
            "manifest": [],
        }
        for v in range(args.n_vehicles)
    ]

    n = len(requests_out)
    lats = np.empty(2 * n + 1)
    lons = np.empty(2 * n + 1)
    lats[0], lons[0] = depot["pt"]["lat"], depot["pt"]["lon"]
    for i, row in reqs.iterrows():
        lats[1 + 2 * i] = row["pickup_lat"]
        lons[1 + 2 * i] = row["pickup_lon"]
        lats[2 + 2 * i] = row["dropoff_lat"]
        lons[2 + 2 * i] = row["dropoff_lon"]
    matrix = haversine_m(lats[:, None], lons[:, None], lats[None, :], lons[None, :]) / ASSUMED_SPEED_MPS

    manifest = {
        "requests": requests_out,
        "depot": depot,
        "driver_runs": driver_runs,
        "travel_time_matrix": matrix.tolist(),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest))
    print(f"Wrote {len(requests_out)} requests, {len(driver_runs)} empty vehicles -> {out_path}")
    print(f"Vehicle window: 0 - {vehicle_end_time}s | request pickups: 0 - {int(reqs['t_sec'].max())}s")


if __name__ == "__main__":
    main()
