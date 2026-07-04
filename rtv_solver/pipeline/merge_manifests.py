import argparse
import json
from pathlib import Path


def remap_request(req, request_offset, node_offset):
    req = dict(req)
    req["booking_id"] = int(req["booking_id"]) + request_offset

    for key in ["pickup_pt", "dropoff_pt"]:
        req[key] = dict(req[key])
        if "node_id" in req[key] and req[key]["node_id"] is not None:
            req[key]["node_id"] = int(req[key]["node_id"]) + node_offset

    return req


def remap_driver(driver, vehicle_offset, node_offset):
    driver = dict(driver)

    if "state" in driver:
        driver["state"] = dict(driver["state"])
        state = driver["state"]
    else:
        state = driver

    for key in ["run_id", "driver_id"]:
        if key in state:
            state[key] = int(state[key]) + vehicle_offset

    if "loc" in state:
        state["loc"] = dict(state["loc"])
        if "node_id" in state["loc"] and state["loc"]["node_id"] is not None:
            state["loc"]["node_id"] = int(state["loc"]["node_id"]) + node_offset

    if "manifest" in driver:
        new_manifest = []
        for stop in driver["manifest"]:
            stop = dict(stop)
            if "run_id" in stop:
                stop["run_id"] = int(stop["run_id"]) + vehicle_offset
            if "loc" in stop:
                stop["loc"] = dict(stop["loc"])
                if "node_id" in stop["loc"] and stop["loc"]["node_id"] is not None:
                    stop["loc"]["node_id"] = int(stop["loc"]["node_id"]) + node_offset
            new_manifest.append(stop)
        driver["manifest"] = new_manifest

    return driver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = None
    all_requests = []
    all_drivers = []

    request_offset = 0
    vehicle_offset = 0
    node_offset = 0

    for input_path in args.inputs:
        path = Path(input_path)
        data = json.loads(path.read_text())

        if merged is None:
            merged = dict(data)

        requests = data.get("requests", [])
        drivers = data.get("drivers", [])

        for req in requests:
            all_requests.append(
                remap_request(req, request_offset, node_offset)
            )

        for drv in drivers:
            all_drivers.append(
                remap_driver(drv, vehicle_offset, node_offset)
            )

        request_offset += 1000
        vehicle_offset += 1000
        node_offset += 10000

    merged["requests"] = all_requests
    merged["drivers"] = all_drivers

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))

    print(f"Saved merged manifest to {out}")
    print(f"Requests: {len(all_requests)}")
    print(f"Drivers: {len(all_drivers)}")


if __name__ == "__main__":
    main()