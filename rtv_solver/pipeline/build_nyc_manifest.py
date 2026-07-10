"""
Converts a RollingHorizon NYC solver run into a manifest JSON in our own
solver's format (same shape as solutions/li_lim/manifests/*.json), so the
existing GNN training pipeline (train_models_v2.py) can be pointed at it
unmodified.

Inputs:
    - RollingHorizon/data/requests/requests_morning500.csv
      (request_id, pickup_node, pickup_lon, pickup_lat, dropoff_node,
       dropoff_lon, dropoff_lat, time)
    - RollingHorizon/results/<run>/expert_pairs_overlap.csv
      (request_id_a, request_id_b - strict temporally-overlapping expert pairs,
       see RollingHorizon/scripts/label_builder_nyc.py)

What gets synthesized (the raw NYC data has neither time windows nor a
wheelchair flag):
    - pickup_time_window_end   = pickup_time_window_start + MAX_WAITING (300s,
      matches the MAX_WAITING default the RollingHorizon run itself used)
    - dropoff_time_window_start = pickup_time_window_start + haversine-based
      ideal travel time (assumed 6 m/s average urban speed)
    - dropoff_time_window_end   = dropoff_time_window_start + MAX_DETOUR (600s,
      matches RollingHorizon's MAX_DETOUR default)
    - am=1, wc=0 for every request (no group-size/wheelchair info in the raw
      NYC data - this is a placeholder, not a paratransit-specific dataset yet)

travel_time_matrix is filled with haversine distances only to satisfy
NetworkHandler.init_from_payload's requirement that *some* matrix is present
(so it doesn't fall back to a live routing server) - its actual values are
never read by RequestGraphFullBuilder/RequestGraphFeatureBuilder, which work
directly off each Request's own lat/lon/time-window fields instead.

Usage:
    python3 -m rtv_solver.pipeline.build_nyc_manifest \
        --run nyc_morning500_mc3 \
        --out solutions/nyc/manifests/nyc_morning500_mc3.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")

MAX_WAITING = 300   # matches RollingHorizon MAX_WAITING default
MAX_DETOUR = 600    # matches RollingHorizon MAX_DETOUR default
ASSUMED_SPEED_MPS = 6.0  # ~21.6 km/h average urban speed, used only to synthesize
                          # a plausible dropoff time window - not a real routing time


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
    parser.add_argument("--run", default="nyc_morning500_mc3", help="RollingHorizon results/<run> directory name")
    parser.add_argument("--requests-csv", default="data/requests/requests_morning500.csv",
                         help="path (relative to RollingHorizon root) of the request slice used for that run")
    parser.add_argument("--out", default="solutions/nyc/manifests/nyc_morning500_mc3.json",
                         help="output path relative to this (Reduce-then-optimize) repo root")
    args = parser.parse_args()

    requests_csv = RH_ROOT / args.requests_csv
    pairs_csv = RH_ROOT / "results" / args.run / "expert_pairs_overlap.csv"

    reqs = pd.read_csv(
        requests_csv, header=None,
        names=["request_id", "pickup_node", "pickup_lon", "pickup_lat",
               "dropoff_node", "dropoff_lon", "dropoff_lat", "time"],
    )
    reqs["t_sec"] = reqs["time"].apply(parse_time_to_seconds)
    reqs["t_sec"] -= reqs["t_sec"].min()  # anchor to 0 for this slice, matches Li&Lim manifest convention

    travel_time_s = haversine_m(
        reqs["pickup_lat"], reqs["pickup_lon"], reqs["dropoff_lat"], reqs["dropoff_lon"]
    ) / ASSUMED_SPEED_MPS

    requests_out = []
    for i, row in reqs.iterrows():
        pickup_start = int(row["t_sec"])
        pickup_end = pickup_start + MAX_WAITING
        dropoff_start = pickup_start + int(travel_time_s[i])
        dropoff_end = dropoff_start + MAX_DETOUR

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

    # depot: centroid of all pickup points, node_id 0
    depot = {
        "pt": {"lat": float(reqs["pickup_lat"].mean()), "lon": float(reqs["pickup_lon"].mean())},
        "node_id": 0,
    }

    # travel_time_matrix: placeholder only, see module docstring - never read by the
    # request-graph feature/label builders, just needs to be non-None so
    # NetworkHandler.init_from_payload doesn't fall back to a live routing server.
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

    pairs = pd.read_csv(pairs_csv)
    driver_runs = [
        {
            # "state" must be present (any content) so PayloadParser treats this as
            # already-canonical and doesn't try to convert it from the older
            # "Chattanooga" driver format, which requires additional fields we
            # don't have (and don't need - only "manifest[].booking_id" is read
            # by RequestGraphLabelBuilder).
            "state": {"run_id": i, "start_time": 0, "end_time": 0, "am_capacity": 0, "wc_capacity": 0},
            "manifest": [{"booking_id": int(r["request_id_a"])}, {"booking_id": int(r["request_id_b"])}],
        }
        for i, (_, r) in enumerate(pairs.iterrows())
    ]

    manifest = {
        "requests": requests_out,
        "depot": depot,
        "driver_runs": driver_runs,
        "travel_time_matrix": matrix.tolist(),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest))
    print(f"Wrote {len(requests_out)} requests, {len(driver_runs)} expert pairs -> {out_path}")


if __name__ == "__main__":
    main()
