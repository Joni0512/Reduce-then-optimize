"""
Builds a COAML-ready NYC manifest: a single canonical file that serves both as
the solve input (--input_file, manifests get cleared by main.py before solving)
and as the imitation ground truth (--imitation_solution_file, read with
manifests intact by ImitationHandler).

Background: the nyc_dense2000_30min baseline run never reached its intended
30-minute window - results.log shows it stalled and stopped at internal
TIME STAMP 71500 (07:15:00), half-way through, due to exploding per-iteration
ILP assignment time (162s -> 726s between the last two logged steps). So only
requests picked up before ~07:15 have a real, complete (boarding + alighting)
route recorded; anything after that either was never dispatched or has no
alighting event (still "onboard" when the log stops). Of the 859 requests the
baseline did serve, only 408 have both events - see label_builder_nyc.py's
"still onboard at cutoff" count and the RHO-pilot half-window bias check
(2026-07-15 conversation).

This script therefore samples only from those ~408 complete requests, and
documents the effective window as ~15 minutes, not the "30min" the run name
implies.

Unlike build_nyc_manifest.py (label-only, 2-stop pseudo-driver_runs) and
build_nyc_solve_manifest.py (solve-ready, but EMPTY vehicles), this produces
FULL per-vehicle stop sequences reconstructed from the actual observed
boarding/alighting times, in the canonical ManifestEntry shape required by
DriverRun.from_dict / ImitationHandler (run_id, booking_id, order, action,
loc incl. node_id, am, wc, scheduled_time, time_window_start/end, dwell).

Usage:
    python3 -m rtv_solver.pipeline.build_nyc_imitation_manifest \
        --run nyc_dense2000_30min \
        --requests-csv data/requests/requests_dense2000_30min.csv \
        --n-requests 80 --seed 42 \
        --out solutions/nyc/manifests/nyc_imitation80_solve.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")

MAX_WAITING = 300
MAX_DETOUR = 600
ASSUMED_SPEED_MPS = 6.0
VEHICLE_END_TIME_BUFFER = 1200
ACTION_RE = re.compile(r"^([PAR])R(\d+)$")


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


def read_time_hhmmss(x: int) -> int:
    x = int(x)
    hour = (x // 10000) % 100
    minute = (x // 100) % 100
    second = x % 100
    return hour * 3600 + minute * 60 + second


def parse_alighting_times(actions_log: Path) -> dict[int, int]:
    alighting = {}
    with open(actions_log) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            match = ACTION_RE.match(parts[3])
            if not match:
                continue
            action, req_id = match.group(1), int(match.group(2))
            if action == "A":
                alighting[req_id] = read_time_hhmmss(int(parts[1]))
    return alighting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="nyc_dense2000_30min", help="RollingHorizon results/<run> directory name")
    parser.add_argument("--requests-csv", default="data/requests/requests_dense2000_30min.csv")
    parser.add_argument("--n-requests", type=int, default=80)
    parser.add_argument("--am-capacity", type=int, default=3)
    parser.add_argument("--wc-capacity", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="solutions/nyc/manifests/nyc_imitation80_solve.json",
                         help="output path relative to this (Reduce-then-optimize) repo root")
    # 2026-07-16: restrict to a holdout ID list so the imitation reference (and hence the
    # pilot's requests) never overlaps with whatever the GNN checkpoint was trained on.
    parser.add_argument("--request-ids-file", type=str, default=None,
                         help="optional CSV with a 'request_id' column - if set, sample only "
                              "from these IDs (e.g. the holdout split).")
    args = parser.parse_args()

    routes = pd.read_csv(RH_ROOT / "results" / args.run / "routes.csv", sep="\t")
    routes["boarding_time_s"] = routes["boarding_time"].apply(read_time_hhmmss)
    alighting = parse_alighting_times(RH_ROOT / "results" / args.run / "actions.log")
    routes["alighting_time_s"] = routes["request_id"].map(alighting)

    complete = routes.dropna(subset=["alighting_time_s"]).copy()
    complete["alighting_time_s"] = complete["alighting_time_s"].astype(int)
    if args.request_ids_file:
        allowed_ids = set(pd.read_csv(ROOT / args.request_ids_file)["request_id"])
        complete = complete[complete["request_id"].isin(allowed_ids)].reset_index(drop=True)
    print(f"Complete requests available (boarding + alighting both recorded): {len(complete)}")

    if args.n_requests > len(complete):
        raise ValueError(f"--n-requests {args.n_requests} exceeds {len(complete)} available complete requests")
    sample = complete.sample(n=args.n_requests, random_state=args.seed)

    reqs_all = pd.read_csv(
        RH_ROOT / args.requests_csv, header=None,
        names=["request_id", "pickup_node", "pickup_lon", "pickup_lat",
               "dropoff_node", "dropoff_lon", "dropoff_lat", "time"],
    )
    reqs_all["t_sec"] = reqs_all["time"].apply(parse_time_to_seconds)
    window_start = reqs_all["t_sec"].min()  # anchor to the FULL 2000-request window's start (07:00:00)

    reqs = reqs_all[reqs_all["request_id"].isin(sample["request_id"])].copy()
    reqs["t_sec"] -= window_start
    reqs.index = reqs["request_id"].values

    sample = sample.copy()
    sample["boarding_rel_s"] = sample["boarding_time_s"] - window_start
    sample["alighting_rel_s"] = sample["alighting_time_s"] - window_start
    sample = sample.set_index("request_id", drop=False)

    travel_time_s = haversine_m(
        reqs["pickup_lat"], reqs["pickup_lon"], reqs["dropoff_lat"], reqs["dropoff_lon"]
    ) / ASSUMED_SPEED_MPS
    travel_time_s = pd.Series(travel_time_s.values, index=reqs.index)

    node_id_map = {}  # request_id -> (pickup_node_id, dropoff_node_id)
    requests_out = []
    for idx, (rid, row) in enumerate(reqs.sort_values("request_id").iterrows()):
        pickup_start = int(row["t_sec"])
        pickup_end = pickup_start + MAX_WAITING
        dropoff_start = pickup_start + int(travel_time_s[rid])
        dropoff_end = dropoff_start + MAX_DETOUR
        pickup_node_id, dropoff_node_id = 1 + 2 * idx, 2 + 2 * idx
        node_id_map[rid] = (pickup_node_id, dropoff_node_id)

        requests_out.append({
            "booking_id": int(rid),
            "pickup_pt": {"lat": row["pickup_lat"], "lon": row["pickup_lon"], "node_id": pickup_node_id},
            "dropoff_pt": {"lat": row["dropoff_lat"], "lon": row["dropoff_lon"], "node_id": dropoff_node_id},
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

    # Build the real per-vehicle stop sequence: one pickup + one dropoff event
    # per sampled request, sorted by actual observed time within each vehicle.
    stop_events = []
    for rid, row in sample.iterrows():
        vehicle_id = int(row["vehicle_id"])
        pickup_node_id, dropoff_node_id = node_id_map[rid]
        req_row = reqs.loc[rid]
        stop_events.append({
            "vehicle_id": vehicle_id, "booking_id": int(rid), "action": "pickup",
            "time_s": int(row["boarding_rel_s"]), "loc": {"lat": req_row["pickup_lat"], "lon": req_row["pickup_lon"], "node_id": pickup_node_id},
            "window_start": int(req_row["t_sec"]), "window_end": int(req_row["t_sec"]) + MAX_WAITING,
        })
        stop_events.append({
            "vehicle_id": vehicle_id, "booking_id": int(rid), "action": "dropoff",
            "time_s": int(row["alighting_rel_s"]), "loc": {"lat": req_row["dropoff_lat"], "lon": req_row["dropoff_lon"], "node_id": dropoff_node_id},
            "window_start": int(req_row["t_sec"]) + int(travel_time_s[rid]),
            "window_end": int(req_row["t_sec"]) + int(travel_time_s[rid]) + MAX_DETOUR,
        })

    stops_df = pd.DataFrame(stop_events)
    driver_runs = []
    for vehicle_id, group in stops_df.groupby("vehicle_id"):
        group = group.sort_values("time_s").reset_index(drop=True)
        manifest = []
        for order, (_, stop) in enumerate(group.iterrows(), start=1):
            manifest.append({
                "run_id": vehicle_id,
                "booking_id": stop["booking_id"],
                "order": order,
                "action": stop["action"],
                "loc": stop["loc"],
                "am": 1,
                "wc": 0,
                "scheduled_time": int(stop["time_s"]),
                "time_window_start": int(stop["window_start"]),
                "time_window_end": int(stop["window_end"]),
                "dwell": 90,
            })
        max_time = int(group["time_s"].max())
        driver_runs.append({
            "state": {
                "run_id": vehicle_id,
                "start_time": 0,
                "end_time": max_time + VEHICLE_END_TIME_BUFFER,
                "am_capacity": args.am_capacity,
                "wc_capacity": args.wc_capacity,
                "locations_already_serviced": 0,
                "location_dt_seconds": 0,
                "total_locations": len(manifest),
                "loc": {"lat": depot["pt"]["lat"], "lon": depot["pt"]["lon"], "node_id": 0},
            },
            "manifest": manifest,
        })

    n = len(requests_out)
    lats = np.empty(2 * n + 1)
    lons = np.empty(2 * n + 1)
    lats[0], lons[0] = depot["pt"]["lat"], depot["pt"]["lon"]
    for i, r in enumerate(sorted(requests_out, key=lambda x: x["booking_id"])):
        lats[1 + 2 * i] = r["pickup_pt"]["lat"]
        lons[1 + 2 * i] = r["pickup_pt"]["lon"]
        lats[2 + 2 * i] = r["dropoff_pt"]["lat"]
        lons[2 + 2 * i] = r["dropoff_pt"]["lon"]
    matrix = haversine_m(lats[:, None], lons[:, None], lats[None, :], lons[None, :]) / ASSUMED_SPEED_MPS

    manifest_out = {
        "requests": requests_out,
        "depot": depot,
        "driver_runs": driver_runs,
        "travel_time_matrix": matrix.tolist(),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest_out))
    print(f"Wrote {len(requests_out)} requests across {len(driver_runs)} vehicles (real routes) -> {out_path}")
    print("NOTE: effective window is ~15 minutes (07:00-07:15), not the full 30min the source run name implies "
          "- see build_nyc_imitation_manifest.py module docstring.")


if __name__ == "__main__":
    main()
