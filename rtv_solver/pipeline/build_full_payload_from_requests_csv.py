from pathlib import Path
import argparse
import pickle

import pandas as pd


def time_to_seconds(value):
    if pd.isna(value):
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    value = str(value)

    if ":" in value:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    return int(float(value))


def build_full_payload_from_requests_and_vehicles(
    requests_csv_path: str | Path,
    vehicles_csv_path: str | Path,
    output_path: str | Path,
    route_end_time: int = 24 * 3600,
):
    requests_csv_path = Path(requests_csv_path)
    vehicles_csv_path = Path(vehicles_csv_path)
    output_path = Path(output_path)

    requests_df = pd.read_csv(requests_csv_path)
    vehicles_df = pd.read_csv(vehicles_csv_path)

    requests = []

    for _, row in requests_df.iterrows():
        pickup_datetime = pd.to_datetime(row["tpep_pickup_datetime"])

        pickup_time = (
            pickup_datetime.hour * 3600
            + pickup_datetime.minute * 60
            + pickup_datetime.second
        )

        distance = float(row.get("trip_distance", 0.0))

        pickup_dwell = float(row.get("dwell_pickup", 0.0))
        dropoff_dwell = float(row.get("dwell_alight", 0.0))

        # grobe Schätzung: trip_distance in miles -> Sekunden
        # 1 mile ca. 5 Minuten als einfacher Fallback
        estimated_travel_time = max(distance * 300.0, 60.0)

        request = {
            "booking_id": str(row["id"]),
            "am": 1,
            "wc": 0,

            "pickup_time_window_start": pickup_time,
            "pickup_time_window_end": pickup_time + 1800.0,

            "pickup_pt": {
                "lat": float(row["pickup_latitude"]),
                "lon": float(row["pickup_longitude"]),
            },

            "dropoff_time_window_start": pickup_time + estimated_travel_time,
            "dropoff_time_window_end": pickup_time + estimated_travel_time + 1800.0,

            "dropoff_pt": {
                "lat": float(row["dropoff_latitude"]),
                "lon": float(row["dropoff_longitude"]),
            },

            "dwell_pickup": pickup_dwell,
            "dwell_alight": dropoff_dwell,
        }

        requests.append(request)
    
    driver_runs = []
    for _, row in vehicles_df.iterrows():
        start_time = time_to_seconds(row["start_time"])
        capacity = int(row["capacity"])

        driver_runs.append(
            {
                "state": {
                    "run_id": str(row["id"]),
                    "start_time": start_time,
                    "end_time": route_end_time,
                    "am_capacity": capacity,
                    "wc_capacity": 0,
                    "loc_serv": 0,
                    "dt_sec": start_time,
                    "loc": {
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "node_id": int(row["node"]),
                    },
                },
                "manifest": [],
            }
        )

    depot_lat = float(vehicles_df["lat"].median())
    depot_lon = float(vehicles_df["lon"].median())

    payload = {
        "requests": requests,
        "driver_runs": driver_runs,
        "depot": {
            "pt": {
                "lat": depot_lat,
                "lon": depot_lon,
            }
        },
        "travel_time_matrix": None,
        "current_time": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved payload to: {output_path}")
    print(f"Requests: {len(requests)}")
    print(f"Vehicles: {len(driver_runs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--requests-csv", required=True)
    parser.add_argument("--vehicles-csv", required=True)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    build_full_payload_from_requests_and_vehicles(
        requests_csv_path=args.requests_csv,
        vehicles_csv_path=args.vehicles_csv,
        output_path=args.output_path,
    )